"""
CloneWiper - Core Scanning Engine

This module contains all UI-independent core logic, including:
    - File scanning and collection
    - Duplicate file grouping
    - MD5 and perceptual hash calculation
    - SQLite cache management
    - Quick selection strategy calculation

Design Philosophy:
    The core engine is completely independent of UI frameworks and communicates with UI through callback functions.
    This allows the same core logic to be reused by Qt UI or CLI interfaces.

Main Classes:
    - ScanEngine: Main scanning engine class
    - FileItem: File item data class
    - Group: Duplicate file group data class

Author: Mark Yip
Version: 2.0
"""
import os
import stat
import threading
import hashlib
import sqlite3
import time
import atexit
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Callable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
import sys

# Try to import psutil for advanced CPU detection (hybrid architecture support)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional dependencies
try:
    import imagehash  # type: ignore[reportMissingImports]
    from PIL import Image, ImageFile
    IMAGEHASH_AVAILABLE = True
    PIL_AVAILABLE = True
    # Allow loading truncated images (common with some JPEG files)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    try:
        from PIL import Image, ImageFile
        PIL_AVAILABLE = True
        # Allow loading truncated images (common with some JPEG files)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    except ImportError:
        PIL_AVAILABLE = False

if PIL_AVAILABLE:
    Image.MAX_IMAGE_PIXELS = 200000000  # 200 megapixels

# OpenCV for ORB/SSIM/SIFT verification (optional)
# Import at module level and suppress warnings globally
try:
    import cv2
    # Suppress OpenCV warnings globally (10-bit TIFF, RAW files, etc.)
    # Set log level to ERROR (3) to suppress WARN messages
    try:
        cv2.setLogLevel(3)  # 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=SILENT
    except:
        pass  # If setLogLevel fails, continue anyway
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


@dataclass
class FileItem:
    """Represents a single file in a duplicate group."""
    path: str
    size: int
    mtime: float
    file_type: str  # 'image', 'video', 'audio', 'pdf', 'epub'
    metadata: str = ""  # Resolution/duration/bitrate (computed async)
    hash_value: str = ""  # MD5 or pHash


@dataclass
class Group:
    """Represents a group of duplicate files."""
    id: str  # Hash value
    files: List[FileItem]
    total_size: int = 0


class ScanEngine:
    """
    Core scanning and grouping engine (no UI dependencies).
    Uses callbacks for progress reporting instead of direct UI updates.
    """
    
    # File type extensions (same as original)
    IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.svg',
        '.cr2', '.crw', '.nef', '.nrw', '.arw', '.sr2', '.srf', '.srw', '.orf', '.raf',
        '.rw2', '.dng', '.raw', '.pef', '.ptx', '.rwl', '.3fr', '.ari', '.bay', '.cap',
        '.eip', '.iiq', '.cine', '.dcs', '.dcr', '.drf', '.erf', '.fff', '.mef', '.mos',
        '.mrw', '.r3d', '.rwz', '.x3f',
    }
    
    RAW_EXTENSIONS = {
        '.cr2', '.crw', '.nef', '.nrw', '.arw', '.sr2', '.srf', '.srw', '.orf', '.raf',
        '.rw2', '.dng', '.raw', '.pef', '.ptx', '.rwl', '.3fr', '.ari', '.bay', '.cap',
        '.eip', '.iiq', '.cine', '.dcs', '.dcr', '.drf', '.erf', '.fff', '.mef', '.mos',
        '.mrw', '.r3d', '.rwz', '.x3f',
    }
    
    VIDEO_EXTENSIONS = {
        '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg',
        '.3gp', '.3g2', '.ts', '.mts', '.m2ts', '.vob', '.ogv', '.rm', '.rmvb'
    }
    
    AUDIO_EXTENSIONS = {
        '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus', '.alac', '.ape',
        '.aiff', '.mid', '.midi'
    }
    
    PDF_EXTENSIONS = {'.pdf'}
    EPUB_EXTENSIONS = {'.epub', '.mobi', '.azw3'}
    
    # Combined set of all supported extensions
    SUPPORTED_EXTENSIONS = (
        IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | 
        PDF_EXTENSIONS | EPUB_EXTENSIONS
    )
    
    def __init__(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        results_callback: Optional[Callable[[Dict[str, List[str]]], None]] = None,
    ):
        """
        Initialize engine.
        
        Args:
            progress_callback: Called with progress value (0.0-1.0) during scan
            status_callback: Called with status message string
            results_callback: Called with duplicate_groups dict when scan completes
        """
        self.progress_callback = progress_callback or (lambda v: None)
        self.status_callback = status_callback or (lambda s: None)
        self.results_callback = results_callback or (lambda d: None)
        
        # Scan state
        self.scan_cancelled = False
        self.files_scanned = 0
        self.total_files = 0
        
        # Dynamic CPU detection for hybrid architecture (P-cores and E-cores)
        cpu_info = self._detect_cpu_architecture()
        logical_cores = cpu_info['logical_cores']
        physical_cores = cpu_info['physical_cores']
        p_cores = cpu_info.get('p_cores', physical_cores)  # Performance cores
        e_cores = cpu_info.get('e_cores', 0)  # Efficiency cores
        is_hybrid = cpu_info.get('is_hybrid', False)
        
        # Optimize thread pool size based on CPU architecture
        if is_hybrid and p_cores > 0:
            # Hybrid CPU (e.g., Intel 12th/13th gen): Use P-cores for CPU-intensive, all cores for I/O-intensive
            # For I/O-bound tasks (file reading, hash calculation), we can use all cores (P + E)
            # For CPU-bound tasks, prefer P-cores
            # i5-13600: 6 P-cores + 8 E-cores = 14 physical, 20 logical
            # I/O-intensive: Use all logical cores (P + E with hyperthreading)
            # CPU-intensive: Use P-cores * 2 (hyperthreading)
            self.max_workers = min(32, max(8, logical_cores))  # Use all logical cores for I/O
            # Hash calculation is I/O-intensive, use all cores but cap for memory safety
            self.hash_workers = min(32, max(16, logical_cores))
            print(f"DEBUG: Hybrid CPU detected - P-cores: {p_cores}, E-cores: {e_cores}, Logical: {logical_cores}")
            print(f"DEBUG: Using {self.max_workers} workers for general tasks, {self.hash_workers} for hash calculation")
        else:
            # Standard CPU: Use 2x logical cores for I/O-intensive tasks
            cpu_count = logical_cores
            self.max_workers = min(32, max(8, cpu_count * 2))
            self.hash_workers = min(32, max(16, cpu_count * 2))
            print(f"DEBUG: Standard CPU detected - {logical_cores} logical cores")
            print(f"DEBUG: Using {self.max_workers} workers for general tasks, {self.hash_workers} for hash calculation")
        
        self.hash_lock = threading.Lock()
        
        # Batch cache write queue to reduce lock contention
        self._cache_write_queue = []
        self._cache_write_lock = threading.Lock()
        self._cache_write_batch_size = 50  # Batch size for cache writes
        self._cache_write_timer = None
        
        # Batch I/O optimization: Pre-read file data to memory
        # This reduces I/O wait time by prefetching files while hash calculation is in progress
        self._file_prefetch_cache = {}  # file_path -> (file_data_bytes, timestamp)
        self._file_prefetch_lock = threading.Lock()
        # Optimized for large datasets (10K+ images): Increased cache limits
        self._file_prefetch_max_size = 500  # Maximum number of files to cache in memory (up from 100)
        self._file_prefetch_max_bytes = 2048 * 1024 * 1024  # Maximum 2GB cache size (up from 500MB)
        self._file_prefetch_enabled = True  # Enable/disable prefetching
        
        # Results
        self.file_groups: Dict[str, List[str]] = defaultdict(list)
        self.file_groups_raw: Dict[str, List[str]] = {}
        
        # Cache settings
        self.phash_cache_enabled = True
        self.md5_cache_enabled = True
        self.use_imagehash = False
        self.use_multi_hash = False  # True for multi-algorithm, False for single algorithm
        
        # Advanced hash accuracy settings
        # Hamming Distance Threshold:
        #   Hamming distance measures the number of differing bits between two hash values.
        #   For 64-bit perceptual hashes, the maximum distance is 64 (completely different).
        #   - Threshold of 4 means up to 4 bits can differ (93.75% similarity required)
        #   - Lower values (1-3): Very strict matching, very few false positives, may miss some duplicates
        #   - Higher values (5-12): More lenient matching, catches more duplicates, may include false positives
        #   Example: If hash1 = "a1b2c3d4" and hash2 = "a1b2c3d5", they differ by 1 bit (distance = 1)
        self.hamming_threshold = 4  # Hamming distance threshold (set to 4 for stricter matching)
        self.min_agreement = 3  # Minimum algorithms that must agree (default: 3 out of 4)
        self.hash_image_size = 512  # Image size for hash calculation (increased to 512 for better accuracy)
        
        self.use_opencv_verification = False  # Use OpenCV for final verification (False/None, 'ssim', 'orb', 'sift')
        self.opencv_verification_method = None  # 'ssim', 'orb', or 'sift' (None = disabled)
        
        # Deferred ORB verification: For large datasets, optionally defer ORB to deletion time
        # DEFAULT: False - since users review thumbnails anyway, ORB during scan filters false positives earlier
        # Set to True only if skipping visual review and want faster scan at cost of manual verification later
        self.defer_orb_verification = False  # Default: False (verify during scan)
        
        # ORB GPU acceleration settings
        # CUDA support disabled (using CPU only)
        self.use_orb_gpu = False
        
        # ORB descriptor cache for faster verification
        self._orb_cache = {}  # file_path -> (keypoints, descriptors)
        self._orb_cache_lock = threading.Lock()
        self._orb_cache_max_size = 1000  # Maximum cached ORB descriptors to prevent unbounded growth
        
        # Persistent cache DB
        self._phash_db = None
        self._phash_db_lock = threading.Lock()
        self._phash_db_path = self._get_default_phash_db_path()
        # Debug: Always log the database path being used
        print(f"DEBUG: Cache database path: {self._phash_db_path}")
        self._cache_db_ro_local = None
        
        # Cache stats
        self._phash_stats_lock = threading.Lock()
        self._phash_cache_lookups = 0
        self._phash_cache_hits = 0
        self._phash_cache_misses = 0
        self._phash_cache_puts = 0
        self._md5_cache_lookups = 0
        self._md5_cache_hits = 0
        self._md5_cache_misses = 0
        self._md5_cache_puts = 0
        
        try:
            atexit.register(lambda: self._close_phash_db())
        except Exception:
            pass
    
    def _detect_cpu_architecture(self) -> Dict[str, int]:
        """
        Detect CPU architecture, including hybrid CPUs (P-cores and E-cores).
        
        Returns:
            Dict with keys: logical_cores, physical_cores, p_cores, e_cores, is_hybrid
        """
        logical_cores = os.cpu_count() or 4
        physical_cores = logical_cores
        p_cores = logical_cores
        e_cores = 0
        is_hybrid = False
        
        if PSUTIL_AVAILABLE:
            try:
                # Get physical core count
                physical_cores = psutil.cpu_count(logical=False) or logical_cores
                
                # Try to detect hybrid architecture (Intel 12th/13th gen, Apple M-series)
                # On Windows, we can use CPU model name to detect core types
                try:
                    import platform
                    if platform.system() == 'Windows':
                        import subprocess
                        result = subprocess.run(
                            ['wmic', 'cpu', 'get', 'name'],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if result.returncode == 0:
                            cpu_name = result.stdout.lower()
                            # Intel 12th/13th/14th gen hybrid architecture indicators
                            if any(x in cpu_name for x in ['12th gen', '13th gen', '14th gen', 'intel core i5-13', 'intel core i7-13', 'intel core i9-13', 'intel core i5-14', 'intel core i7-14', 'intel core i9-14']):
                                is_hybrid = True
                                # Common configurations:
                                # i5-13600: 6P + 8E = 14 physical, 20 logical
                                # i7-13700: 8P + 8E = 16 physical, 24 logical
                                # i9-13900: 8P + 16E = 24 physical, 32 logical
                                if 'i5-13' in cpu_name or 'i5-14' in cpu_name:
                                    p_cores = 6
                                    e_cores = 8
                                elif 'i7-13' in cpu_name or 'i7-14' in cpu_name:
                                    p_cores = 8
                                    e_cores = 8
                                elif 'i9-13' in cpu_name or 'i9-14' in cpu_name:
                                    p_cores = 8
                                    e_cores = 16
                                print(f"DEBUG: Detected hybrid CPU by model name - P: {p_cores}, E: {e_cores}, Logical: {logical_cores}")
                    elif platform.system() == 'Darwin':  # macOS
                        # Apple Silicon (M1/M2/M3) are hybrid but handled differently
                        cpu_name = platform.processor().lower()
                        if 'apple' in cpu_name or 'arm' in cpu_name:
                            # Apple Silicon typically has performance and efficiency cores
                            # M1: 4P + 4E, M2: 4P + 4E, M3: 4P + 4E
                            is_hybrid = True
                            p_cores = 4
                            e_cores = 4
                except Exception as e:
                    print(f"DEBUG: CPU model detection error: {e}")
                    
            except Exception as e:
                print(f"DEBUG: CPU detection error: {e}")
        
        return {
            'logical_cores': logical_cores,
            'physical_cores': physical_cores,
            'p_cores': p_cores,
            'e_cores': e_cores,
            'is_hybrid': is_hybrid
        }
    
    def _get_system_dirs(self) -> set:
        """Get platform-specific system directories to skip during scanning."""
        import platform
        system = platform.system()
        if system == "Windows":
            return {
                'windows', '$recycle.bin', 'system volume information',
                'recovery', 'program files', 'program files (x86)'
            }
        elif system == "Darwin":  # macOS
            return {
                '.ds_store', 'library', 'system', 'applications',
                'users', 'volumes', 'cores', 'private', 'usr', 'bin',
                'sbin', 'opt', 'dev', 'etc', 'tmp', 'var', 'net'
            }
        else:
            return set()
    
    def _get_default_phash_db_path(self) -> str:
        """Choose a writable cache location (cross-platform).
        
        Always uses platform-specific cache directories to ensure EXE and main.py use the same database.
        """
        import platform
        
        # Always try platform-specific cache directories first (same for EXE and main.py)
        try:
            system = platform.system()
            if system == "Windows":
                base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
                if base:
                    cache_dir = os.path.join(base, "CloneWiper")
                    os.makedirs(cache_dir, exist_ok=True)
                    db_path = os.path.join(cache_dir, "phash_cache.sqlite3")
                    # Debug: Log the database path being used
                    if hasattr(self, '_debug_mode') and self._debug_mode:
                        print(f"DEBUG: Using cache DB path: {db_path}")
                    return db_path
            elif system == "Darwin":  # macOS
                cache_dir = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "CloneWiper")
                os.makedirs(cache_dir, exist_ok=True)
                db_path = os.path.join(cache_dir, "phash_cache.sqlite3")
                if hasattr(self, '_debug_mode') and self._debug_mode:
                    print(f"DEBUG: Using cache DB path: {db_path}")
                return db_path
        except Exception as e:
            # Log error but continue to fallback
            print(f"DEBUG: Failed to use platform cache directory: {e}")
        
        # Fallback: Use user's home directory (works for both EXE and main.py)
        try:
            home = os.path.expanduser("~")
            if home:
                cache_dir = os.path.join(home, ".CloneWiper")
                os.makedirs(cache_dir, exist_ok=True)
                db_path = os.path.join(cache_dir, "phash_cache.sqlite3")
                print(f"DEBUG: Using fallback cache DB path: {db_path}")
                return db_path
        except Exception as e:
            print(f"DEBUG: Fallback to home directory failed: {e}")
        
        # Last resort: current directory (not recommended, but ensures it works)
        try:
            db_path = os.path.abspath("phash_cache.sqlite3")
            print(f"DEBUG: Using last resort cache DB path: {db_path}")
            return db_path
        except Exception:
            return "phash_cache.sqlite3"
    
    def _ensure_phash_db(self):
        """Lazy-init SQLite DB (thread-safe)."""
        if self._phash_db is not None:
            return
        with self._phash_db_lock:
            if self._phash_db is not None:
                return
            try:
                conn = sqlite3.connect(self._phash_db_path, timeout=30, check_same_thread=False)
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                except Exception:
                    pass
                
                # Check if table exists and has correct structure
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='phash_cache'")
                table_exists = cursor.fetchone() is not None
                
                if table_exists:
                    # Check PRIMARY KEY structure
                    cursor.execute("PRAGMA table_info(phash_cache)")
                    columns = cursor.fetchall()
                    pk_columns = [col[1] for col in columns if col[5] == 1]  # col[5] is pk flag
                    
                    # If PRIMARY KEY is only (path), need to migrate
                    if len(pk_columns) == 1 and pk_columns[0] == 'path':
                        try:
                            print("DEBUG: Migrating phash_cache table structure from PRIMARY KEY (path) to PRIMARY KEY (path, algo)...")
                            # Step 1: Create new table
                            cursor.execute("""
                                CREATE TABLE phash_cache_new (
                                    path TEXT NOT NULL,
                                    size INTEGER NOT NULL,
                                    mtime_ns INTEGER NOT NULL,
                                    algo TEXT NOT NULL,
                                    hash TEXT NOT NULL,
                                    updated INTEGER NOT NULL,
                                    PRIMARY KEY (path, algo)
                                )
                            """)
                            # Step 2: Copy data
                            cursor.execute("""
                                INSERT INTO phash_cache_new (path, size, mtime_ns, algo, hash, updated)
                                SELECT path, size, mtime_ns, algo, hash, updated
                                FROM phash_cache
                            """)
                            migrated_count = cursor.rowcount
                            # Step 3: Drop old table
                            cursor.execute("DROP TABLE phash_cache")
                            # Step 4: Rename new table
                            cursor.execute("ALTER TABLE phash_cache_new RENAME TO phash_cache")
                            # Step 5: Recreate indexes
                            cursor.execute("CREATE INDEX IF NOT EXISTS idx_phash_updated ON phash_cache(updated)")
                            cursor.execute("CREATE INDEX IF NOT EXISTS idx_phash_path_algo ON phash_cache(path, algo)")
                            conn.commit()
                            print(f"DEBUG: Migration completed successfully. Migrated {migrated_count} entries.")
                        except Exception as migrate_error:
                            print(f"DEBUG: Migration failed: {migrate_error}")
                            conn.rollback()
                            # Continue with old structure - backward compatibility will handle it
                else:
                    # Table doesn't exist, create with correct structure
                    conn.execute(
                        """
                        CREATE TABLE phash_cache (
                            path TEXT NOT NULL,
                            size INTEGER NOT NULL,
                            mtime_ns INTEGER NOT NULL,
                            algo TEXT NOT NULL,
                            hash TEXT NOT NULL,
                            updated INTEGER NOT NULL,
                            PRIMARY KEY (path, algo)
                        )
                        """
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_phash_updated ON phash_cache(updated)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_phash_path_algo ON phash_cache(path, algo)")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS md5_cache (
                        path TEXT PRIMARY KEY,
                        size INTEGER NOT NULL,
                        mtime_ns INTEGER NOT NULL,
                        hash TEXT NOT NULL,
                        updated INTEGER NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_md5_updated ON md5_cache(updated)")
                conn.commit()
                self._phash_db = conn
            except Exception:
                self._phash_db = None
                self.phash_cache_enabled = False
                self.md5_cache_enabled = False
    
    def _get_cache_db_ro(self):
        """Get a per-thread read-only SQLite connection for cache lookups."""
        try:
            loc = getattr(self, "_cache_db_ro_local", None)
            if loc is None:
                loc = threading.local()
                self._cache_db_ro_local = loc
        except Exception:
            loc = None
        try:
            conn = getattr(loc, "conn", None) if loc is not None else None
        except Exception:
            conn = None
        if conn is not None:
            return conn
        self._ensure_phash_db()
        if getattr(self, "_phash_db", None) is None:
            return None
        try:
            conn = sqlite3.connect(self._phash_db_path, timeout=30, check_same_thread=False)
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA temp_store=MEMORY;")
            except Exception:
                pass
        except Exception:
            return None
        try:
            if loc is not None:
                loc.conn = conn
        except Exception:
            pass
        return conn
    
    def _close_phash_db(self):
        try:
            with self._phash_db_lock:
                if self._phash_db is not None:
                    try:
                        self._phash_db.close()
                    except Exception:
                        pass
                    self._phash_db = None
        except Exception:
            pass
    
    def _phash_cache_get(self, file_path: str, size: int, mtime_ns: int, algo: str) -> Optional[str]:
        """Return cached hash string or None."""
        if not self.phash_cache_enabled:
            return None
        conn = self._get_cache_db_ro()
        if conn is None:
            return None
        try:
            p = os.path.abspath(file_path)
        except Exception:
            p = file_path
        try:
            # Query cache with path, algo, size, and mtime_ns to ensure exact match
            # This ensures we only return cached hash if file hasn't been modified
            row = conn.execute(
                "SELECT hash FROM phash_cache WHERE path=? AND algo=? AND size=? AND mtime_ns=?",
                (p, str(algo), int(size), int(mtime_ns)),
            ).fetchone()
            
            # If not found, try legacy algorithm names for backward compatibility
            if not row:
                legacy_algo = None
                if algo == "single_hash":
                    legacy_algo = "average_hash"
                elif algo == "video_single_hash":
                    legacy_algo = "video_average_hash"
                
                if legacy_algo:
                    row = conn.execute(
                        "SELECT hash FROM phash_cache WHERE path=? AND algo=? AND size=? AND mtime_ns=?",
                        (p, legacy_algo, int(size), int(mtime_ns)),
                    ).fetchone()
        except Exception as e:
            # Log cache query errors for debugging (only in debug mode)
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"DEBUG: Cache query error for {os.path.basename(file_path)}: {e}")
            return None
        if not row:
            return None
        try:
            h0 = row[0]
            return str(h0)
        except Exception as e:
            # Log cache result parsing errors for debugging
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"DEBUG: Cache result parsing error for {os.path.basename(file_path)}: {e}")
            return None
    
    def _phash_cache_put(self, file_path: str, size: int, mtime_ns: int, algo: str, hash_str: str):
        if not self.phash_cache_enabled:
            return
        self._ensure_phash_db()
        if self._phash_db is None:
            return
        try:
            p = os.path.abspath(file_path)
        except Exception:
            p = file_path
        now = int(time.time())
        with self._phash_db_lock:
            try:
                self._phash_db.execute(
                    """
                    INSERT INTO phash_cache(path, size, mtime_ns, algo, hash, updated)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(path, algo) DO UPDATE SET
                      size=excluded.size,
                      mtime_ns=excluded.mtime_ns,
                      hash=excluded.hash,
                      updated=excluded.updated
                    """,
                    (p, int(size), int(mtime_ns), str(algo), str(hash_str), now),
                )
                self._phash_db.commit()
            except Exception:
                pass
    
    def _md5_cache_get(self, file_path: str, size: int, mtime_ns: int) -> Optional[str]:
        """Return cached MD5 string or None."""
        if not self.md5_cache_enabled:
            return None
        conn = self._get_cache_db_ro()
        if conn is None:
            return None
        try:
            p = os.path.abspath(file_path)
        except Exception:
            p = file_path
        try:
            row = conn.execute(
                "SELECT size, mtime_ns, hash FROM md5_cache WHERE path=?",
                (p,),
            ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        try:
            s0, m0, h0 = row
            if int(s0) == int(size) and int(m0) == int(mtime_ns):
                return str(h0)
        except Exception:
            return None
        return None
    
    def _md5_cache_put(self, file_path: str, size: int, mtime_ns: int, hash_str: str):
        if not self.md5_cache_enabled:
            return
        self._ensure_phash_db()
        if self._phash_db is None:
            return
        try:
            p = os.path.abspath(file_path)
        except Exception:
            p = file_path
        now = int(time.time())
        with self._phash_db_lock:
            try:
                self._phash_db.execute(
                    """
                    INSERT INTO md5_cache(path, size, mtime_ns, hash, updated)
                    VALUES(?,?,?,?,?)
                    ON CONFLICT(path) DO UPDATE SET
                      size=excluded.size,
                      mtime_ns=excluded.mtime_ns,
                      hash=excluded.hash,
                      updated=excluded.updated
                    """,
                    (p, int(size), int(mtime_ns), str(hash_str), now),
                )
                self._phash_db.commit()
            except Exception:
                pass
    
    def is_image_file(self, file_path: str) -> bool:
        return os.path.splitext(file_path.lower())[1] in self.IMAGE_EXTENSIONS
    
    def is_raw_image_file(self, file_path: str) -> bool:
        return os.path.splitext(file_path.lower())[1] in self.RAW_EXTENSIONS
    
    def is_video_file(self, file_path: str) -> bool:
        return os.path.splitext(file_path.lower())[1] in self.VIDEO_EXTENSIONS
    
    def is_audio_file(self, file_path: str) -> bool:
        return os.path.splitext(file_path.lower())[1] in self.AUDIO_EXTENSIONS
    
    def is_pdf_file(self, file_path: str) -> bool:
        return os.path.splitext(file_path.lower())[1] in self.PDF_EXTENSIONS
    
    def is_epub_file(self, file_path: str) -> bool:
        return os.path.splitext(file_path.lower())[1] in self.EPUB_EXTENSIONS
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if file extension is in the supported list."""
        ext = os.path.splitext(file_path.lower())[1]
        return ext in self.SUPPORTED_EXTENSIONS
    
    def _calculate_single_hash(self, img: Image.Image) -> Optional[str]:
        """
        Calculate single perceptual hash using phash (perceptual hash) algorithm.
        phash is more accurate than average_hash and is a good balance between speed and accuracy.
        
        Pre-processing optimization:
        - Pre-resize large images to 256x256 before hash calculation
        - This improves accuracy by preserving more detail while keeping computation efficient
        - 256x256 matches multi-hash mode, allowing phash reuse between single and multi-hash modes
        
        Args:
            img: PIL Image object
            
        Returns:
            Hash string or None on error
        """
        try:
            if not IMAGEHASH_AVAILABLE:
                return None
            
            # Optimization: Check mode before converting (avoid unnecessary conversion)
            if img.mode != 'RGB':
                try:
                    img = img.convert('RGB')
                except Exception as convert_error:
                    print(f"DEBUG: Image mode conversion failed, using original mode: {convert_error}")
            
            # Pre-resize large images to improve accuracy
            # 256x256 matches multi-hash mode, allowing phash reuse between single and multi-hash modes
            # This helps phash capture more accurate frequency features during DCT
            # Optimization: Use BILINEAR instead of LANCZOS for faster resizing (2-3x faster, negligible accuracy impact for hash)
            try:
                w, h = img.size
                if w > 256 or h > 256:
                    # Resize to 256x256 max (maintain aspect ratio)
                    # BILINEAR is faster than LANCZOS and sufficient for hash calculation
                    img.thumbnail((256, 256), Image.Resampling.BILINEAR)
            except Exception as resize_error:
                # If resize fails, continue with original image
                print(f"DEBUG: Image resize failed in single hash, using original: {resize_error}")
            
            # Use phash (perceptual hash) - more accurate than average_hash
            try:
                return str(imagehash.phash(img))
            except Exception as e:
                print(f"DEBUG: phash failed: {e}")
                # Fallback to average_hash if phash fails
                try:
                    return str(imagehash.average_hash(img))
                except Exception:
                    return None
        except Exception as e:
            print(f"DEBUG: Single hash calculation failed: {e}")
            return None
    
    def _calculate_multi_hash(self, img: Image.Image, cached_phash: Optional[str] = None) -> Optional[str]:
        """
        Calculate combined perceptual hash using multiple algorithms for better accuracy.
        Uses: average_hash, phash (perceptual), dhash (difference), and whash (wavelet).
        
        Optimizations:
        - Pre-resize large images to 256x256 before hash calculation for better accuracy
        - This improves hash quality by preserving more detail while keeping computation efficient
        - 256x256 is optimal for multi-hash: balances accuracy and performance for 4 algorithms
        - Reuse cached phash from single-hash mode if available (both use 256x256 now)
        - Convert to RGB once and reuse
        
        Args:
            img: PIL Image object
            cached_phash: Optional cached phash from single-hash mode to reuse
            
        Returns:
            Combined hash string in format: "avg_phash_diff_wave" or None on error
        """
        try:
            if not IMAGEHASH_AVAILABLE:
                return None
            
            # Optimization: Check mode before converting (avoid unnecessary conversion)
            if img.mode != 'RGB':
                try:
                    img = img.convert('RGB')
                except Exception as convert_error:
                    # If conversion fails, try to continue with original mode
                    print(f"DEBUG: Image mode conversion failed, using original mode: {convert_error}")
            
            # Pre-resize large images to improve accuracy for multi-hash calculation
            # 256x256 is optimal: provides better detail than 128x128 for multiple algorithms
            # while still being computationally efficient for parallel hash calculation
            # Optimization: Use BILINEAR instead of LANCZOS for faster resizing (2-3x faster, negligible accuracy impact)
            try:
                w, h = img.size
                if w > 256 or h > 256:
                    # Resize to 256x256 max (maintain aspect ratio)
                    # BILINEAR is faster than LANCZOS and sufficient for hash calculation
                    img.thumbnail((256, 256), Image.Resampling.BILINEAR)
            except Exception as resize_error:
                # If resize fails, continue with original image
                print(f"DEBUG: Image resize failed in multi hash, using original: {resize_error}")
            
            # Optimization: Direct calculation instead of ThreadPoolExecutor for CPU-bound tasks
            # ThreadPoolExecutor overhead (thread creation, context switching) can be slower than direct calls
            # for CPU-intensive hash calculations. Direct calls are faster for small operations.
            hash_results = {}
            
            # Calculate all hashes directly (faster than ThreadPoolExecutor for CPU-bound tasks)
            try:
                hash_results['average'] = str(imagehash.average_hash(img))
            except Exception as e:
                print(f"DEBUG: average_hash failed: {e}")
                hash_results['average'] = None
            
            # Reuse cached phash if available (from single-hash mode, both use 256x256 now)
            if cached_phash:
                hash_results['perceptual'] = cached_phash
            else:
                try:
                    hash_results['perceptual'] = str(imagehash.phash(img))
                except Exception as e:
                    print(f"DEBUG: phash failed: {e}")
                    hash_results['perceptual'] = None
            
            try:
                hash_results['difference'] = str(imagehash.dhash(img))
            except Exception as e:
                print(f"DEBUG: dhash failed: {e}")
                hash_results['difference'] = None
            
            try:
                hash_results['wavelet'] = str(imagehash.whash(img))
            except Exception as e:
                print(f"DEBUG: whash failed: {e}")
                hash_results['wavelet'] = None
            
            # Combine available hashes (use placeholder for failed ones)
            hashes = [
                hash_results.get('average', ''),
                hash_results.get('perceptual', ''),
                hash_results.get('difference', ''),
                hash_results.get('wavelet', ''),
            ]
            
            # If at least one hash succeeded, return combined hash
            if any(hashes):
                # Return as string with separator for compatibility
                # Individual hashes will be compared using hamming distance in grouping phase
                combined_hash = "_".join(hashes)
                return combined_hash
            else:
                # All hashes failed
                return None
            
        except Exception as e:
            print(f"DEBUG: Multi-hash calculation failed: {e}")
            # Fallback to single hash if multi-hash fails
            try:
                return str(imagehash.average_hash(img))
            except Exception:
                return None
    
    def _prefetch_file_data(self, file_path: str) -> Optional[bytes]:
        """
        Pre-fetch file data to memory for faster I/O.
        Uses async I/O to read file while other operations are in progress.
        
        Args:
            file_path: Path to file to prefetch
            
        Returns:
            File data as bytes, or None if failed
        """
        if not self._file_prefetch_enabled:
            return None
        
        try:
            # Check cache first
            with self._file_prefetch_lock:
                if file_path in self._file_prefetch_cache:
                    data, _ = self._file_prefetch_cache[file_path]
                    return data
            
            # Check file size before reading (avoid reading huge files)
            try:
                file_size = os.path.getsize(file_path)
                # Only prefetch files smaller than 50MB to avoid memory issues
                if file_size > 50 * 1024 * 1024:
                    return None
            except Exception:
                return None
            
            # Read file data
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Add to cache (with size limit)
                with self._file_prefetch_lock:
                    # Remove oldest entries if cache is too large
                    cache_size = sum(len(d) for d, _ in self._file_prefetch_cache.values())
                    while (len(self._file_prefetch_cache) >= self._file_prefetch_max_size or 
                           cache_size + len(data) > self._file_prefetch_max_bytes):
                        if not self._file_prefetch_cache:
                            break
                        # Remove oldest entry (FIFO)
                        oldest_key = next(iter(self._file_prefetch_cache))
                        old_data, _ = self._file_prefetch_cache.pop(oldest_key)
                        cache_size -= len(old_data)
                    
                    # Add new entry
                    self._file_prefetch_cache[file_path] = (data, time.time())
                
                return data
            except Exception as e:
                print(f"DEBUG: File prefetch failed for {os.path.basename(file_path)}: {e}")
                return None
        except Exception:
            return None
    
    def _prefetch_files_batch(self, file_paths: List[str], max_workers: int = 4):
        """
        Pre-fetch multiple files in parallel using async I/O.
        
        Args:
            file_paths: List of file paths to prefetch
            max_workers: Number of parallel prefetch workers
        """
        if not self._file_prefetch_enabled or not file_paths:
            return
        
        def prefetch_one(file_path: str):
            """Prefetch a single file."""
            try:
                self._prefetch_file_data(file_path)
            except Exception:
                pass  # Silently fail for prefetch
        
        # Prefetch in parallel batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all prefetch tasks
            futures = [executor.submit(prefetch_one, fp) for fp in file_paths[:self._file_prefetch_max_size * 2]]
            # Don't wait for completion - let them run in background
            # This allows hash calculation to proceed while files are being prefetched
    
    def _load_image_from_prefetch(self, file_path: str) -> Optional['Image.Image']:
        """
        Load image from prefetch cache if available, otherwise load normally.
        
        Args:
            file_path: Path to image file
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            # Try to get from prefetch cache
            with self._file_prefetch_lock:
                if file_path in self._file_prefetch_cache:
                    data, _ = self._file_prefetch_cache[file_path]
                    # Load image from bytes
                    import io
                    try:
                        img = Image.open(io.BytesIO(data))
                        # Verify image is actually valid by trying to access size
                        _ = img.size
                        return img
                    except Exception:
                        # Invalid image data in cache, remove it
                        self._file_prefetch_cache.pop(file_path, None)
                        return None
        except Exception:
            pass
        
        # Fallback to normal file loading
        try:
            img = Image.open(file_path)
            # Verify image is actually valid by trying to access size
            _ = img.size
            return img
        except Exception:
            # Image cannot be identified or is corrupted - this is expected for some files
            return None
    
    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate file hash (multi-algorithm pHash for images/videos if enabled, else MD5)."""
        try:
            if self.use_imagehash and IMAGEHASH_AVAILABLE:
                file_ext = os.path.splitext(file_path)[1].lower()
                common_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
                is_raw_file = file_ext in self.RAW_EXTENSIONS
                is_video_file = file_ext in self.VIDEO_EXTENSIONS
                
                # Handle video files with perceptual hashing
                if is_video_file:
                    return self._calculate_video_perceptual_hash(file_path)
                
                if file_ext in common_image_extensions or is_raw_file:
                    try:
                        st = os.stat(file_path)
                        size = int(getattr(st, "st_size", 0) or 0)
                        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
                    except Exception:
                        st = None
                        size = 0
                        mtime_ns = 0
                    
                    algo = "multi_hash" if self.use_multi_hash else "single_hash"  # Algorithm name based on mode
                    cached_phash = None  # For reusing phash from single-hash cache in multi-hash mode
                    
                    if st is not None:
                        try:
                            with self._phash_stats_lock:
                                self._phash_cache_lookups += 1
                        except Exception:
                            pass
                        
                        # First, try to get cached hash for current algorithm
                        cached = self._phash_cache_get(file_path, size, mtime_ns, algo)
                        if cached:
                            try:
                                with self._phash_stats_lock:
                                    self._phash_cache_hits += 1
                            except Exception:
                                pass
                            return f"img_{cached}"
                        
                        # If multi-hash mode and no multi-hash cache, try to reuse phash from single-hash cache
                        if self.use_multi_hash:
                            cached_single = self._phash_cache_get(file_path, size, mtime_ns, "single_hash")
                            if cached_single:
                                # Reuse the phash from single-hash cache (both use 256x256 now)
                                cached_phash = cached_single
                                try:
                                    with self._phash_stats_lock:
                                        self._phash_cache_hits += 1  # Count as cache hit for phash reuse
                                except Exception:
                                    pass
                        
                        if not cached and not cached_phash:
                            try:
                                with self._phash_stats_lock:
                                    self._phash_cache_misses += 1
                            except Exception:
                                pass
                            # Debug: Log cache miss for multi_hash to help diagnose
                            if self.use_multi_hash and self._phash_cache_misses % 100 == 0:
                                print(f"DEBUG: Cache miss for multi_hash: {os.path.basename(file_path)} (algo={algo}, size={size}, mtime_ns={mtime_ns})")
                    
                    try:
                        img = None
                        if is_raw_file:
                            # Handle RAW files using rawpy
                            try:
                                import rawpy
                                with rawpy.imread(file_path) as raw:
                                    try:
                                        # Try to extract embedded thumbnail first (fast)
                                        thumb = raw.extract_thumb()
                                        if thumb.format == rawpy.ThumbFormat.JPEG:
                                            import io
                                            img = Image.open(io.BytesIO(thumb.data))
                                        else:
                                            # Bitmap thumbnail
                                            img = Image.fromarray(thumb.data)
                                    except Exception:
                                        # Fallback to full postprocess (slower but more accurate)
                                        rgb = raw.postprocess(
                                            use_camera_wb=True,
                                            no_auto_bright=True,
                                            bright=1.0,
                                            half_size=True  # Use half size for faster processing
                                        )
                                        img = Image.fromarray(rgb)
                            except ImportError:
                                # rawpy not available, fall back to MD5
                                pass
                            except Exception as raw_error:
                                # RAW processing failed, fall back to MD5
                                # Only log if it's not a common "unsupported format" error
                                # rawpy errors can be bytes or strings
                                error_str = ""
                                if isinstance(raw_error, bytes):
                                    try:
                                        # Decode bytes to get actual error message
                                        error_str = raw_error.decode('utf-8', errors='ignore')
                                    except Exception:
                                        error_str = str(raw_error)
                                else:
                                    error_str = str(raw_error)
                                
                                # Silently ignore common "unsupported format" errors
                                # Check the decoded/string error message (case-insensitive)
                                error_lower = error_str.lower()
                                if ('unsupported file format' not in error_lower and 
                                    'not raw file' not in error_lower):
                                    # Only log unexpected errors
                                    print(f"DEBUG: RAW file processing failed for {os.path.basename(file_path)}: {raw_error}")
                                # Silently fall back to MD5 for unsupported formats
                                pass
                        else:
                            # Handle common image formats
                            # ImageFile.LOAD_TRUNCATED_IMAGES is already set to True at module level
                            # This allows loading truncated images (common with some JPEG files)
                            try:
                                # Try to load from prefetch cache first (faster I/O)
                                img = self._load_image_from_prefetch(file_path)
                                if img is None:
                                    # Fallback to normal file loading
                                    try:
                                        img = Image.open(file_path)
                                        # Verify image is valid
                                        _ = img.size
                                    except Exception:
                                        # Image cannot be identified or is corrupted
                                        # This is expected for some files, silently fall back to MD5
                                        img = None
                                
                                if img is not None:
                                    # Optimization: For large images, use thumbnail loading for faster processing
                                    # This is especially beneficial for multi-hash calculation
                                    if self.use_multi_hash:
                                        # Get image size without fully loading (fast operation)
                                        w, h = img.size
                                        max_dimension = self.hash_image_size  # Use configurable size (default: 256, can be 512 for better accuracy)
                                        if w > max_dimension or h > max_dimension:
                                            # For large images: thumbnail is MUCH faster than processing full size
                                            # Optimization: Use BILINEAR instead of LANCZOS for faster resizing
                                            # BILINEAR is 2-3x faster with negligible accuracy impact for hash calculation
                                            # The thumbnail operation is in-place and memory-efficient
                                            img.thumbnail((max_dimension, max_dimension), Image.Resampling.BILINEAR)
                                            # Note: thumbnail() is optimized and only loads what's needed for the final size
                                        else:
                                            # Small image: no resizing needed, just load normally
                                            img.load()
                                    else:
                                        # Single hash mode - load normally (no resizing needed for single hash)
                                        img.load()
                            except Exception as img_open_error:
                                # If loading fails, try opening without load() call
                                # Some truncated images can still be processed
                                try:
                                    img = Image.open(file_path)
                                    # Verify image is valid
                                    _ = img.size
                                    # Apply thumbnail optimization if multi-hash
                                    if self.use_multi_hash:
                                        w, h = img.size
                                        max_dimension = self.hash_image_size  # Use configurable size
                                        if w > max_dimension or h > max_dimension:
                                            # Optimization: Use BILINEAR for faster resizing
                                            img.thumbnail((max_dimension, max_dimension), Image.Resampling.BILINEAR)
                                except Exception:
                                    # Image cannot be opened or identified - this is expected for corrupted files
                                    # Silently fall back to MD5 without logging (reduces noise)
                                    img = None
                        
                        if img is not None:
                            # Use single or multi-hash algorithm based on setting
                            if self.use_multi_hash:
                                # Pass cached phash to reuse from single-hash mode
                                img_hash = self._calculate_multi_hash(img, cached_phash=cached_phash)
                            else:
                                img_hash = self._calculate_single_hash(img)
                            img.close()  # Explicitly close to free memory
                            
                            if st is not None and img_hash:
                                self._phash_cache_put(file_path, size, mtime_ns, algo, img_hash)
                                try:
                                    with self._phash_stats_lock:
                                        self._phash_cache_puts += 1
                                except Exception:
                                    pass
                            return f"img_{img_hash}" if img_hash else None
                        else:
                            # Image could not be loaded (corrupted or unsupported format)
                            # Silently fall back to MD5 - this is expected for some files
                            pass
                    except Exception as img_error:
                        # Image processing failed, fall back to MD5
                        # Only log unexpected errors (not common "cannot identify" errors)
                        error_str = str(img_error)
                        if 'cannot identify' not in error_str.lower() and 'unsupported' not in error_str.lower():
                            print(f"DEBUG: Image hash calculation failed for {os.path.basename(file_path)}: {img_error}")
                        pass
            
            return self.calculate_file_hash_cpu(file_path)
        except Exception as e:
            print(f"DEBUG: calculate_file_hash error {file_path}: {e}")
            return None
    
    def _calculate_video_perceptual_hash(self, file_path: str) -> Optional[str]:
        """Calculate perceptual hash for video files by extracting keyframes."""
        try:
            # Check cache first
            try:
                st = os.stat(file_path)
                size = int(getattr(st, "st_size", 0) or 0)
                mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
            except Exception:
                st = None
                size = 0
                mtime_ns = 0
            
            algo = "video_multi_hash" if self.use_multi_hash else "video_single_hash"  # Algorithm name based on mode
            if st is not None:
                try:
                    with self._phash_stats_lock:
                        self._phash_cache_lookups += 1
                except Exception:
                    pass
                cached = self._phash_cache_get(file_path, size, mtime_ns, algo)
                if cached:
                    try:
                        with self._phash_stats_lock:
                            self._phash_cache_hits += 1
                    except Exception:
                        pass
                    return f"vid_{cached}"
                else:
                    try:
                        with self._phash_stats_lock:
                            self._phash_cache_misses += 1
                    except Exception:
                        pass
            
            # Try to import OpenCV for video processing
            try:
                import cv2
            except ImportError:
                # OpenCV not available, fall back to MD5
                print(f"DEBUG: OpenCV not available for video hash, falling back to MD5 for {os.path.basename(file_path)}")
                # Return None to trigger MD5 fallback in calculate_file_hash
                return None
            
            # Extract keyframes from video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                cap.release()
                return None
            
            try:
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps <= 0 or frame_count <= 0:
                    cap.release()
                    return None
                
                duration = frame_count / fps
                
                # Extract frames at multiple time points for better accuracy
                # Sample at: start (0%), 25%, 50%, 75%, and end (100%)
                time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
                frame_hashes = []
                
                for time_ratio in time_points:
                    # Calculate frame number
                    target_frame = int(frame_count * time_ratio)
                    # Clamp to valid range
                    target_frame = max(0, min(target_frame, frame_count - 1))
                    
                    # Seek to target frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        try:
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Resize frame to reduce processing time (max 512px width)
                            height, width = frame_rgb.shape[:2]
                            if width > 512:
                                scale = 512 / width
                                new_width = 512
                                new_height = int(height * scale)
                                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                            
                            # Convert to PIL Image and calculate multi-hash
                            img = Image.fromarray(frame_rgb)
                            # Use multi-hash algorithm for each frame
                            if self.use_multi_hash:
                                frame_hash = self._calculate_multi_hash(img)
                            else:
                                frame_hash = self._calculate_single_hash(img)
                            if frame_hash:
                                frame_hashes.append(frame_hash)
                            img.close()
                        except Exception as frame_error:
                            print(f"DEBUG: Failed to process frame at {time_ratio*100}% for {file_path}: {frame_error}")
                            continue
                
                cap.release()
                
                if not frame_hashes:
                    # No frames extracted, fall back to MD5
                    return None
                
                # Combine frame hashes into a single video hash
                # Each frame hash is already a multi-hash, so we combine them with separator
                video_hash = "_".join(frame_hashes)
                
                # Cache the result
                if st is not None and video_hash:
                    self._phash_cache_put(file_path, size, mtime_ns, algo, video_hash)
                    try:
                        with self._phash_stats_lock:
                            self._phash_cache_puts += 1
                    except Exception:
                        pass
                
                return f"vid_{video_hash}"
                
            except Exception as video_error:
                cap.release()
                print(f"DEBUG: Video hash calculation failed for {file_path}: {video_error}")
                return None
                
        except Exception as e:
            print(f"DEBUG: _calculate_video_perceptual_hash error for {file_path}: {e}")
            return None
    
    def calculate_file_hash_cpu(self, file_path: str) -> Optional[str]:
        """Calculate MD5 hash using CPU."""
        try:
            st = os.stat(file_path)
            file_size = int(getattr(st, "st_size", 0) or 0)
            try:
                mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
            except Exception:
                mtime_ns = int(getattr(st, "st_mtime", 0.0) * 1e9)
            
            if self.md5_cache_enabled:
                try:
                    with self._phash_stats_lock:
                        self._md5_cache_lookups += 1
                except Exception:
                    pass
                try:
                    cached = self._md5_cache_get(file_path, file_size, mtime_ns)
                    if cached:
                        with self._phash_stats_lock:
                            self._md5_cache_hits += 1
                        return cached
                    else:
                        with self._phash_stats_lock:
                            self._md5_cache_misses += 1
                except Exception:
                    pass
            
            md5_hash = hashlib.md5()
            
            # Adaptive strategy: preload small files to RAM for speed, chunk large files for memory efficiency
            # Threshold: 10MB - files smaller than this are preloaded, larger files are chunked
            PRELOAD_THRESHOLD = 10 * 1024 * 1024  # 10MB
            
            if file_size <= PRELOAD_THRESHOLD:
                # Preload small files to RAM for faster I/O
                with open(file_path, "rb") as f:
                    data = f.read()
                    md5_hash.update(data)
            else:
                # Chunk large files to avoid memory issues
                chunk_size = 262144 if file_size > 100 * 1024 * 1024 else 131072
                with open(file_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(chunk_size), b""):
                        md5_hash.update(byte_block)
            
            digest = md5_hash.hexdigest()
            
            if digest and self.md5_cache_enabled:
                self._md5_cache_put(file_path, file_size, mtime_ns, digest)
                with self._phash_stats_lock:
                    self._md5_cache_puts += 1
            return digest
        except Exception as e:
            print(f"DEBUG: calculate_file_hash_cpu error {file_path}: {e}")
            return None
    
    def calculate_partial_hash(self, file_path: str, chunk_size: int = 4096) -> Optional[str]:
        """Calculate partial MD5 hash (Start + Middle + End) with caching.
        
        The partial hash is cached to avoid recalculating on every scan.
        Uses a special cache key format: "partial_{hash}" in MD5 cache.
        """
        try:
            st = os.stat(file_path)
            file_size = int(getattr(st, "st_size", 0) or 0)
            if file_size <= 0:
                return None
            
            try:
                mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
            except Exception:
                mtime_ns = int(getattr(st, "st_mtime", 0.0) * 1e9)
            
            # Check cache first (reuse MD5 cache infrastructure with special marker)
            # We use a virtual path with "partial_" prefix to distinguish from full MD5
            if self.md5_cache_enabled:
                try:
                    with self._phash_stats_lock:
                        self._md5_cache_lookups += 1
                except Exception:
                    pass
                try:
                    # Use a special cache key for partial hash
                    cache_path = f"partial_{os.path.abspath(file_path)}"
                    cached = self._md5_cache_get(cache_path, file_size, mtime_ns)
                    if cached:
                        with self._phash_stats_lock:
                            self._md5_cache_hits += 1
                        return cached
                    else:
                        with self._phash_stats_lock:
                            self._md5_cache_misses += 1
                except Exception:
                    pass
            
            # Calculate partial hash
            with open(file_path, 'rb') as f:
                data = f.read(chunk_size)
                if not data:
                    return None
                if file_size > chunk_size * 3:
                    f.seek(file_size // 2)
                    data += f.read(chunk_size)
                    f.seek(-chunk_size, 2)
                    data += f.read(chunk_size)
                partial_hash = hashlib.md5(data).hexdigest()
            
            # Cache the partial hash result
            if self.md5_cache_enabled and partial_hash:
                try:
                    cache_path = f"partial_{os.path.abspath(file_path)}"
                    self._md5_cache_put(cache_path, file_size, mtime_ns, partial_hash)
                    with self._phash_stats_lock:
                        self._md5_cache_puts += 1
                except Exception:
                    pass
            
            return partial_hash
        except Exception:
            return None
    
    def collect_files_by_size(
        self,
        path: str,
        size_groups: Dict[int, List[str]],
        total_collected: List[int],
        size_groups_lock: threading.Lock,
        total_collected_lock: threading.Lock,
        total_image_files: List[int],
        total_image_files_lock: threading.Lock,
        all_image_files: Optional[List[str]] = None,
        all_image_files_lock: Optional[threading.Lock] = None,
    ):
        """Collect files and group by size (optimized with os.scandir)."""
        try:
            update_counter = 0
            local_size_groups = defaultdict(list)
            local_count = 0
            local_image_count = 0
            local_all_images = []
            
            stack = [path]
            system_dirs_lower = self._get_system_dirs()
            
            while stack:
                if self.scan_cancelled:
                    return
                
                current_path = stack.pop()
                
                try:
                    with os.scandir(current_path) as it:
                        for entry in it:
                            if self.scan_cancelled:
                                return
                            
                            try:
                                name_lower = entry.name.lower()
                                
                                if entry.is_dir(follow_symlinks=False):
                                    if name_lower not in system_dirs_lower:
                                        stack.append(entry.path)
                                        
                                elif entry.is_file(follow_symlinks=False):
                                    if entry.name.startswith('._'):
                                        continue
                                    
                                    # Only process supported file types (images, videos, audio, PDFs, EPUBs)
                                    if not self.is_supported_file(entry.name):
                                        continue
                                    
                                    stat_info = entry.stat(follow_symlinks=False)
                                    file_size = stat_info.st_size
                                    
                                    if file_size > 0:
                                        local_size_groups[file_size].append(entry.path)
                                        local_count += 1
                                        
                                        if self.is_image_file(entry.name):
                                            local_image_count += 1
                                            if all_image_files is not None:
                                                local_all_images.append(entry.path)
                                        
                                        update_counter += 1
                                        
                                        if update_counter >= 5000:
                                            with size_groups_lock:
                                                for size, paths in local_size_groups.items():
                                                    size_groups[size].extend(paths)
                                                local_size_groups.clear()
                                            
                                            with total_collected_lock:
                                                total_collected[0] += local_count
                                                local_count = 0
                                            
                                            with total_image_files_lock:
                                                total_image_files[0] += local_image_count
                                                local_image_count = 0
                                            
                                            if all_image_files is not None and all_image_files_lock is not None and local_all_images:
                                                with all_image_files_lock:
                                                    all_image_files.extend(local_all_images)
                                                local_all_images = []
                                            
                                            update_counter = 0
                            except (OSError, PermissionError):
                                continue
                except (OSError, PermissionError):
                    continue
            
            if local_size_groups:
                with size_groups_lock:
                    for size, paths in local_size_groups.items():
                        size_groups[size].extend(paths)
            
            if all_image_files is not None and all_image_files_lock is not None and local_all_images:
                with all_image_files_lock:
                    all_image_files.extend(local_all_images)
            
            with total_collected_lock:
                total_collected[0] += local_count
            
            with total_image_files_lock:
                total_image_files[0] += local_image_count
        except Exception as e:
            print(f"collect_files_by_size error: {e}")
    
    def perform_partial_hashing(self, file_paths: List[str], phase_num: int = None, total_phases: int = None) -> Dict[str, List[str]]:
        """Perform partial hashing for fast pre-filtering.
        
        Args:
            file_paths: List of file paths to process
            phase_num: Phase number for status display (if None, will be determined from use_imagehash)
            total_phases: Total number of phases for status display (if None, will be determined from use_imagehash)
        """
        partial_groups = defaultdict(list)
        failed_count = 0
        total = len(file_paths)
        processed = 0
        
        # Determine phase info if not provided
        if phase_num is None or total_phases is None:
            if self.use_imagehash:
                phase_num = 2
                total_phases = 3
            else:
                phase_num = 3
                total_phases = 4
        
        def _partial_key(fp: str) -> Optional[str]:
            try:
                st = os.stat(fp)
                size = int(getattr(st, "st_size", 0) or 0)
                if size <= 0:
                    return None
                ph = self.calculate_partial_hash(fp)
                if not ph:
                    return None
                return f"{size}_{ph}"
            except Exception:
                return None
        
        # Partial hashing is I/O-intensive, use more workers
        max_workers = max(4, int(self.hash_workers * 1.5))  # Use hash_workers as base
        max_inflight = max(64, max_workers * 8)  # Increase inflight tasks for better parallelism
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            it = iter(file_paths)
            inflight = {}
            
            def submit_one():
                try:
                    fp = next(it)
                except StopIteration:
                    return False
                if self.scan_cancelled:
                    return False
                fut = executor.submit(_partial_key, fp)
                inflight[fut] = fp
                return True
            
            for _ in range(min(max_inflight, total if total else max_inflight)):
                if not submit_one():
                    break
            
            # Optimized progress update interval (aligned with full hash calculation):
            # - For small batches (< 100 files): update every file (real-time)
            # - For medium batches (100-1000 files): update every 10 files
            # - For large batches (> 1000 files): update every 1% or every 50 files, whichever is more frequent
            if total < 100:
                progress_update_interval = 1  # Every file for small batches
            elif total < 1000:
                progress_update_interval = 10  # Every 10 files for medium batches
            else:
                # For large batches, update every 1% or every 50 files (whichever is smaller)
                progress_update_interval = min(max(1, total // 100), 50)
            
            # Time-based throttling: update at most once per 100ms to avoid UI overload
            last_update_time = time.time()
            min_update_interval = 0.1  # 100ms minimum between updates
            last_ui = 0
            
            while inflight:
                if self.scan_cancelled:
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    return partial_groups
                
                done = set()
                try:
                    for fut in as_completed(list(inflight.keys()), timeout=0.2):
                        done.add(fut)
                        break
                except Exception:
                    done = set()
                
                if not done:
                    continue
                
                for fut in done:
                    fp = inflight.pop(fut, None)
                    try:
                        key = fut.result()
                        if key:
                            partial_groups[key].append(fp)
                        else:
                            failed_count += 1
                    except Exception:
                        failed_count += 1
                    
                    processed += 1
                    
                    # Update progress with throttling (aligned with full hash calculation logic)
                    current_time = time.time()
                    should_update = (
                        processed % progress_update_interval == 0 or 
                        processed == total or
                        (current_time - last_update_time) >= min_update_interval
                    )
                    
                    if should_update:
                        # Calculate phase progress (0.0-1.0) for current phase only
                        phase_progress = processed / total if total > 0 else 0.0
                        # Pass phase progress (0.0-1.0) instead of overall progress
                        self.progress_callback(phase_progress)
                        self.status_callback(f"{phase_num}/{total_phases} Scanning: Pre-filtering (Partial Hash) {processed:,}/{total:,}")
                        last_update_time = current_time
                        last_ui = processed
                    
                    while len(inflight) < max_inflight and submit_one():
                        pass
        
        return partial_groups
    
    def _group_by_similarity_multi_hash(self, hash_groups: Dict[str, List[str]], skip_orb_verification: bool = False) -> Dict[str, List[str]]:
        """
        Group files by multi-algorithm hash similarity using hamming distance.
        
        Uses a voting mechanism: requires at least 3 out of 4 algorithms to agree
        that images are similar (hamming distance <= threshold).
        
        This reduces false positives compared to exact match while maintaining
        good accuracy for true duplicates.
        
        Args:
            hash_groups: Dict mapping hash strings to file lists
            skip_orb_verification: If True, skip ORB verification (will be done in separate phase)
            
        Returns:
            Dict mapping group IDs to file lists (duplicate groups)
        """
        if not IMAGEHASH_AVAILABLE:
            # Fallback to exact match if imagehash not available
            duplicate_groups = {}
            for hash_val, files in hash_groups.items():
                if len(files) > 1:
                    unique_files = list(dict.fromkeys(files))
                    if len(unique_files) > 1:
                        duplicate_groups[hash_val] = unique_files
            return duplicate_groups
        
        # Hamming Distance Threshold Explanation:
        #   Hamming distance = number of bits that differ between two hash values
        #   For 64-bit perceptual hashes (average_hash, phash, dhash, whash):
        #   - Maximum distance: 64 (completely different images)
        #   - Distance 0: Identical hashes (exact duplicates)
        #   - Distance 1-8: Very similar images (likely duplicates)
        #   - Distance 9-16: Similar images (may be duplicates with modifications)
        #   - Distance >16: Different images
        #   
        #   Threshold of 8 means: "If two hashes differ by 8 bits or less, consider them similar"
        #   This allows for minor image modifications (resize, compression, slight edits)
        #   while still detecting duplicates.
        HAMMING_THRESHOLD = self.hamming_threshold
        
        # Voting mechanism: require at least N out of 4 algorithms to agree
        # Use configurable agreement from instance (default: 3)
        MIN_AGREEMENT = self.min_agreement
        
        # Parse multi-hash strings and group by similarity
        hash_to_files = {}  # hash_string -> [files]
        for hash_val, files in hash_groups.items():
            if hash_val and isinstance(hash_val, str) and '_' in hash_val:
                hash_to_files[hash_val] = files
        
        if not hash_to_files:
            return {}
        
        # Build similarity groups with optimizations
        duplicate_groups = {}
        processed = set()
        
        hash_list = list(hash_to_files.items())
        total_hashes = len(hash_list)
        
        # Optimization: Pre-parse all hashes to avoid repeated parsing
        parsed_hashes = {}  # hash_string -> (avg, phash, dhash, whash) or None
        for hash_val, files in hash_list:
            try:
                parts = hash_val.split('_')
                if len(parts) == 4:
                    parsed_hashes[hash_val] = tuple(parts)
                else:
                    parsed_hashes[hash_val] = None
            except Exception:
                parsed_hashes[hash_val] = None
        
        # Optimization: Early exit for exact matches (fast path)
        # First, handle exact matches (these are definitely duplicates)
        # Note: Even exact hash matches should be verified with ORB if enabled,
        # as hash collisions can occur (though rare)
        exact_match_groups = {}
        for hash_val, files in hash_list:
            if hash_val not in processed and len(files) > 1:
                unique_files = list(dict.fromkeys(files))
                if len(unique_files) > 1:
                    # If ORB verification is enabled, verify all files pairwise
                    # This ensures that files that are similar to each other (but not to the first file) are still grouped together
                    # Note: ORB verification will be done in a separate phase if skip_orb_verification is True
                    if self.opencv_verification_method and not skip_orb_verification:
                        print(f"DEBUG: Exact hash match found for {len(unique_files)} files, verifying with ORB (pairwise comparison)...")
                        
                        # Use union-find approach to group similar files
                        # Each file starts in its own group
                        file_to_group = {f: {f} for f in unique_files}
                        
                        # Compare all pairs of files
                        for i, file1 in enumerate(unique_files):
                            for file2 in unique_files[i+1:]:
                                print(f"DEBUG: Verifying exact match pair: {os.path.basename(file1)} <-> {os.path.basename(file2)}")
                                if self._verify_similarity_opencv(hash_val, hash_val, [file1], [file2], method=self.opencv_verification_method):
                                    print(f"DEBUG: ORB verification passed for pair")
                                    # Merge groups
                                    group1 = file_to_group[file1]
                                    group2 = file_to_group[file2]
                                    merged_group = group1 | group2
                                    for f in merged_group:
                                        file_to_group[f] = merged_group
                                else:
                                    print(f"DEBUG: ORB verification failed for pair")
                        
                        # Find the largest group (or all groups with at least 2 files)
                        groups = {}
                        seen_files = set()
                        for file, group in file_to_group.items():
                            if file not in seen_files:
                                group_id = min(group)  # Use minimum file path as group ID
                                if group_id not in groups:
                                    groups[group_id] = sorted(list(group))
                                seen_files.update(group)
                        
                        # Create groups for all sets with at least 2 files
                        group_count = 0
                        for group_id, group_files in groups.items():
                            if len(group_files) >= 2:
                                # Use a unique key for each subgroup
                                subgroup_key = f"{hash_val}_subgroup_{group_count}"
                                exact_match_groups[subgroup_key] = group_files
                                print(f"DEBUG: Exact match subgroup created with {len(group_files)} verified files: {[os.path.basename(f) for f in group_files]}")
                                group_count += 1
                        
                        if group_count == 0:
                            print(f"DEBUG: Exact match group rejected: no pairs passed ORB verification")
                    else:
                        # No ORB verification, use all files
                        exact_match_groups[hash_val] = unique_files
                    processed.add(hash_val)
        
        # Two-phase filtering: Phase 1 - Quick filter using average_hash only
        # This reduces the candidate set before expensive multi-algorithm comparison
        remaining_hashes = [(h, f) for h, f in hash_list if h not in processed]
        
        if len(remaining_hashes) > 100:
            # Phase 1: Quick filter using average_hash (fastest algorithm)
            # Build average_hash index for fast lookup
            avg_hash_index = {}  # avg_hash_value -> [hash_strings]
            for hash_val, files in remaining_hashes:
                parsed = parsed_hashes.get(hash_val)
                if parsed and parsed[0]:  # parsed[0] is average_hash
                    try:
                        avg_hash = imagehash.hex_to_hash(parsed[0])
                        # Create buckets for similar average hashes
                        # Use hash value as key (simplified, but effective)
                        avg_key = str(avg_hash)
                        if avg_key not in avg_hash_index:
                            avg_hash_index[avg_key] = []
                        avg_hash_index[avg_key].append((hash_val, files, avg_hash))
                    except Exception:
                        pass
            
            # Phase 1: LSH-based similarity search (O(n) average case)
            # Build LSH buckets using bit sampling
            # Phase 1: Find candidates using LSH (Locality-Sensitive Hashing)
            print(f"DEBUG: Building LSH index for {len(remaining_hashes)} hashes...")
            lsh_buckets = defaultdict(list)
            num_bands = 8
            bits_per_band = 8
            
            for hash_val, files in remaining_hashes:
                parsed = parsed_hashes.get(hash_val)
                if not parsed or not parsed[1]:
                    continue
                try:
                    phash = imagehash.hex_to_hash(parsed[1])
                    avg_hash = imagehash.hex_to_hash(parsed[0]) if parsed[0] else None
                    hash_bits = phash.hash.flatten()
                    for band_idx in range(num_bands):
                        start_bit = (band_idx * bits_per_band) % 64
                        sampled_bits = [int(hash_bits[(start_bit + i * 7) % 64]) for i in range(bits_per_band)]
                        bucket_key = (band_idx, int(''.join(map(str, sampled_bits)), 2))
                        lsh_buckets[bucket_key].append((hash_val, files, avg_hash, phash))
                except Exception:
                    pass
            
            print(f"DEBUG: LSH created {len(lsh_buckets)} buckets")
            candidate_pairs = set()
            for bucket_key, bucket_items in lsh_buckets.items():
                if len(bucket_items) < 2:
                    continue
                for i, (hash1, files1, h_avg1, h_phash1) in enumerate(bucket_items):
                    for j in range(i + 1, len(bucket_items)):
                        hash2, files2, h_avg2, h_phash2 = bucket_items[j]
                        if hash1 != hash2:
                            try:
                                phash_dist = h_phash1 - h_phash2 if h_phash1 and h_phash2 else 999
                                if phash_dist <= HAMMING_THRESHOLD * 2:
                                    if hash1 < hash2:
                                        candidate_pairs.add((hash1, hash2))
                                    else:
                                        candidate_pairs.add((hash2, hash1))
                            except Exception:
                                pass
            
            # Phase 2: Detailed comparison only for candidate pairs
            # Use parallel processing for detailed comparison
            def compare_pair_internal(pair):
                """Compare a pair of hashes using all algorithms."""
                hash1, hash2 = pair
                if hash1 in processed or hash2 in processed:
                    return None
                
                parsed1 = parsed_hashes.get(hash1)
                parsed2 = parsed_hashes.get(hash2)
                if not parsed1 or not parsed2:
                    return None
                
                avg1, phash1, dhash1, whash1 = parsed1
                avg2, phash2, dhash2, whash2 = parsed2
                
                try:
                    # Pre-convert all hashes
                    h_avg1 = imagehash.hex_to_hash(avg1) if avg1 else None
                    h_phash1 = imagehash.hex_to_hash(phash1) if phash1 else None
                    h_dhash1 = imagehash.hex_to_hash(dhash1) if dhash1 else None
                    h_whash1 = imagehash.hex_to_hash(whash1) if whash1 else None
                    
                    h_avg2 = imagehash.hex_to_hash(avg2) if avg2 else None
                    h_phash2 = imagehash.hex_to_hash(phash2) if phash2 else None
                    h_dhash2 = imagehash.hex_to_hash(dhash2) if dhash2 else None
                    h_whash2 = imagehash.hex_to_hash(whash2) if whash2 else None
                    
                    agreements = 0
                    distances = {}
                    
                    # Check all algorithms
                    if h_avg1 and h_avg2:
                        dist = h_avg1 - h_avg2
                        distances['avg'] = dist
                        if dist <= HAMMING_THRESHOLD:
                            agreements += 1
                    if h_phash1 and h_phash2:
                        dist = h_phash1 - h_phash2
                        distances['phash'] = dist
                        if dist <= HAMMING_THRESHOLD:
                            agreements += 1
                    if h_dhash1 and h_dhash2:
                        dist = h_dhash1 - h_dhash2
                        distances['dhash'] = dist
                        if dist <= HAMMING_THRESHOLD:
                            agreements += 1
                    if h_whash1 and h_whash2:
                        dist = h_whash1 - h_whash2
                        distances['whash'] = dist
                        if dist <= HAMMING_THRESHOLD:
                            agreements += 1
                    
                    files1 = hash_to_files.get(hash1, [])
                    files2 = hash_to_files.get(hash2, [])
                    if agreements >= MIN_AGREEMENT:
                        print(f"DEBUG: Hash similarity match: {os.path.basename(files1[0]) if files1 else 'N/A'} <-> {os.path.basename(files2[0]) if files2 else 'N/A'}")
                        print(f"DEBUG:   Agreements: {agreements}/{4}, distances: {distances}, threshold: {HAMMING_THRESHOLD}, min_agreement: {MIN_AGREEMENT}")
                        # Optional OpenCV verification for higher accuracy
                        # Skip ORB if deferred (for large datasets, verify only on deletion)
                        should_verify_now = self.opencv_verification_method and not skip_orb_verification and not self.defer_orb_verification
                        if should_verify_now:
                            files1 = hash_to_files.get(hash1, [])
                            files2 = hash_to_files.get(hash2, [])
                            print(f"DEBUG: ORB verification called for pair: {os.path.basename(files1[0]) if files1 else 'N/A'} <-> {os.path.basename(files2[0]) if files2 else 'N/A'}")
                            verification_result = self._verify_similarity_opencv(hash1, hash2, files1, files2, method=self.opencv_verification_method)
                            print(f"DEBUG: ORB verification result: {verification_result}")
                            if not verification_result:
                                print(f"DEBUG: ORB verification failed, rejecting pair")
                                return None  # Verification failed
                            print(f"DEBUG: ORB verification passed")
                        elif self.defer_orb_verification and self.opencv_verification_method:
                            print(f"DEBUG: ORB verification deferred (will verify on deletion)")
                        return (hash1, hash2)
                    return None
                except Exception:
                    return None
            
            # Parallel comparison of candidate pairs
            similarity_pairs = []
            if candidate_pairs:
                # Limit candidate pairs to avoid excessive processing
                MAX_CANDIDATE_PAIRS = min(50000, len(candidate_pairs))  # Cap at 50K pairs
                candidate_list = list(candidate_pairs)[:MAX_CANDIDATE_PAIRS]
                print(f"DEBUG: Found {len(candidate_pairs)} candidate pairs, processing {len(candidate_list)} pairs")
                if self.opencv_verification_method:
                    print(f"DEBUG: ORB verification enabled, will verify {len(candidate_list)} candidate pairs")
                
                # Update status if ORB verification is enabled
                total_pairs = len(candidate_list)
                processed_pairs = [0]
                last_update_time = [time.time()]
                
                def compare_pair_with_progress(pair):
                    """Compare pair and update progress."""
                    result = compare_pair_internal(pair)
                    processed_pairs[0] += 1
                    
                    # Update progress every 100 pairs or every 0.5 seconds
                    # Note: Progress updates are skipped if ORB verification is deferred to separate phase
                    if not skip_orb_verification and (processed_pairs[0] % 100 == 0 or (time.time() - last_update_time[0]) >= 0.5):
                        if self.opencv_verification_method:
                            status_msg = f"Verifying {processed_pairs[0]:,}/{total_pairs:,} pairs with {self.opencv_verification_method.upper()}..."
                            self.status_callback(status_msg)
                            print(f"DEBUG: {status_msg}")
                        else:
                            status_msg = f"Comparing {processed_pairs[0]:,}/{total_pairs:,} hash pairs..."
                            self.status_callback(status_msg)
                        last_update_time[0] = time.time()
                    
                    return result
                
                # Use ThreadPoolExecutor for parallel comparison
                with ThreadPoolExecutor(max_workers=min(8, self.max_workers)) as executor:
                    results = executor.map(compare_pair_with_progress, candidate_list)
                    for result in results:
                        if result:
                            similarity_pairs.append(result)
            
            # Build groups from similarity pairs
            # Use union-find approach for grouping
            hash_to_group = {}  # hash -> set of hashes in same group
            for hash1, hash2 in similarity_pairs:
                if hash1 not in hash_to_group:
                    hash_to_group[hash1] = {hash1}
                if hash2 not in hash_to_group:
                    hash_to_group[hash2] = {hash2}
                # Merge groups
                group1 = hash_to_group[hash1]
                group2 = hash_to_group[hash2]
                merged = group1 | group2
                for h in merged:
                    hash_to_group[h] = merged
            
            # Create duplicate groups
            seen_groups = set()
            for hash_val, group in hash_to_group.items():
                group_id = min(group)  # Use minimum hash as group ID
                if group_id in seen_groups:
                    continue
                seen_groups.add(group_id)
                
                # Collect all files from this group
                all_files = []
                for h in group:
                    if h in hash_to_files:
                        all_files.extend(hash_to_files[h])
                
                if len(all_files) > 1:
                    unique_files = list(dict.fromkeys(all_files))
                    if len(unique_files) > 1:
                        duplicate_groups[group_id] = unique_files
                        processed.update(group)
        
        else:
            # For small datasets, use original sequential approach (simpler and faster)
            # More aggressive comparison limit for small datasets
            MAX_COMPARISONS = min(500, total_hashes - 1)  # Reduced from 1000 to 500
            
            for i, (hash1, files1) in enumerate(remaining_hashes):
                if hash1 in processed:
                    continue
                
                parsed1 = parsed_hashes.get(hash1)
                if parsed1 is None:
                    continue
                
                avg1, phash1, dhash1, whash1 = parsed1
                
                # Pre-convert hashes once for this hash1
                try:
                    h_avg1 = imagehash.hex_to_hash(avg1) if avg1 else None
                    h_phash1 = imagehash.hex_to_hash(phash1) if phash1 else None
                    h_dhash1 = imagehash.hex_to_hash(dhash1) if dhash1 else None
                    h_whash1 = imagehash.hex_to_hash(whash1) if whash1 else None
                except Exception:
                    continue
                
                # Find similar hashes
                similar_files = list(files1)
                
                comparisons_made = 0
                for j, (hash2, files2) in enumerate(remaining_hashes[i+1:], start=i+1):
                    if hash2 in processed or comparisons_made >= MAX_COMPARISONS:
                        break
                    
                    comparisons_made += 1
                    
                    parsed2 = parsed_hashes.get(hash2)
                    if parsed2 is None:
                        continue
                    
                    avg2, phash2, dhash2, whash2 = parsed2
                    
                    # Two-phase: Quick check with average_hash first
                    agreements = 0
                    if h_avg1 and avg2:
                        try:
                            h_avg2 = imagehash.hex_to_hash(avg2)
                            dist = h_avg1 - h_avg2
                            if dist <= HAMMING_THRESHOLD:
                                agreements += 1
                            elif dist > HAMMING_THRESHOLD * 2:
                                # Early exit: very different, skip detailed comparison
                                continue
                        except Exception:
                            pass
                    
                    # Detailed comparison only if average_hash was close
                    if agreements > 0 or h_avg1 is None:
                        if h_phash1 and phash2:
                            try:
                                h_phash2 = imagehash.hex_to_hash(phash2)
                                if h_phash1 - h_phash2 <= HAMMING_THRESHOLD:
                                    agreements += 1
                            except Exception:
                                pass
                        
                        if h_dhash1 and dhash2:
                            try:
                                h_dhash2 = imagehash.hex_to_hash(dhash2)
                                if h_dhash1 - h_dhash2 <= HAMMING_THRESHOLD:
                                    agreements += 1
                            except Exception:
                                pass
                        
                        if h_whash1 and whash2:
                            try:
                                h_whash2 = imagehash.hex_to_hash(whash2)
                                if h_whash1 - h_whash2 <= HAMMING_THRESHOLD:
                                    agreements += 1
                            except Exception:
                                pass
                    
                    if agreements >= MIN_AGREEMENT:
                        print(f"DEBUG: Hash similarity match (small dataset path): {os.path.basename(similar_files[0]) if similar_files else 'N/A'} <-> {os.path.basename(files2[0]) if files2 else 'N/A'}")
                        print(f"DEBUG:   Agreements: {agreements}/{4}, threshold: {HAMMING_THRESHOLD}, min_agreement: {MIN_AGREEMENT}")
                        # Optional OpenCV verification for higher accuracy
                        # Skip ORB if deferred (for large datasets, verify only on deletion)
                        should_verify_now = self.opencv_verification_method and not skip_orb_verification and not self.defer_orb_verification
                        if should_verify_now:
                            print(f"DEBUG: ORB verification called (small dataset path)")
                            if not self._verify_similarity_opencv(hash1, hash2, similar_files, files2, method=self.opencv_verification_method):
                                print(f"DEBUG: ORB verification failed (small dataset path), skipping match")
                                continue  # Verification failed, skip this match
                            print(f"DEBUG: ORB verification passed (small dataset path)")
                        elif self.defer_orb_verification and self.opencv_verification_method:
                            print(f"DEBUG: ORB verification deferred (small dataset path)")
                        similar_files.extend(files2)
                        processed.add(hash2)
                
                if len(similar_files) > 1:
                    unique_files = list(dict.fromkeys(similar_files))
                    if len(unique_files) > 1:
                        duplicate_groups[hash1] = unique_files
                
                processed.add(hash1)
        
        # Merge exact match groups with similarity groups
        duplicate_groups.update(exact_match_groups)
        
        return duplicate_groups
    
    def _verify_similarity_opencv(self, hash1: str, hash2: str, files1: List[str], files2: List[str], method: str = 'ssim') -> bool:
        """
        Verify image similarity using OpenCV methods.
        
        Methods (speed order, fastest to slowest):
        - 'orb': ORB feature matching (~100x faster than SIFT, good accuracy)
        - 'ssim': Structural Similarity Index (medium speed, high accuracy)
        - 'sift': Scale-Invariant Feature Transform (slowest, highest accuracy)
        
        Args:
            hash1, hash2: Hash strings (not used, but kept for consistency)
            files1, files2: File paths to compare
            method: Verification method ('orb', 'ssim', or 'sift')
            
        Returns:
            True if images are similar, False otherwise
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            # OpenCV not available, skip verification
            return True  # Default to accepting if OpenCV not available
        
        try:
            import cv2
            import numpy as np
            
            # Suppress OpenCV warnings for unsupported formats (e.g., 10-bit TIFF, RAW files)
            # Set log level to ERROR (3) to suppress WARN messages before any cv2 operations
            # 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=SILENT
            try:
                original_log_level = cv2.getLogLevel()
            except:
                original_log_level = 2  # Default to WARN if getLogLevel fails
            
            try:
                cv2.setLogLevel(3)  # Set to ERROR level to suppress WARN messages
            except:
                pass  # If setLogLevel fails, continue anyway
            
            try:
                # Get first file from each group for comparison
                file1 = files1[0] if files1 else None
                file2 = files2[0] if files2 else None
                
                if not file1 or not file2 or not os.path.exists(file1) or not os.path.exists(file2):
                    print(f"DEBUG: OpenCV {method} verification skipped: files not found or invalid")
                    return True  # Default to accepting if files don't exist
                
                # Read images - try OpenCV first, fallback to PIL/rawpy for unsupported formats
                # OpenCV may fail for: 10-bit TIFF, RAW formats (ARW, CR2, NEF, etc.)
                # Use safe imread to handle Windows path encoding issues
                # Suppress OpenCV warnings globally for this verification operation
                try:
                    cv2.setLogLevel(3)  # ERROR level to suppress WARN messages
                except:
                    pass
                
                img1 = self._imread_safe(file1, cv2.IMREAD_GRAYSCALE)
                img2 = self._imread_safe(file2, cv2.IMREAD_GRAYSCALE)
                
                # If OpenCV failed, try alternative methods
                if img1 is None:
                    img1 = self._load_image_alternative(file1)
                    if img1 is None:
                        # Skip this pair if we can't read the image
                        return True  # Default to accepting if images can't be read
                
                if img2 is None:
                    img2 = self._load_image_alternative(file2)
                    if img2 is None:
                        # Skip this pair if we can't read the image
                        return True  # Default to accepting if images can't be read
                
                if img1 is None or img2 is None:
                    return True  # Default to accepting if images can't be read
                
                # Resize to same size for comparison (use smaller dimension)
                # Validate images before resizing
                if img1 is None or img2 is None or len(img1.shape) < 2 or len(img2.shape) < 2:
                    print(f"DEBUG: Invalid images before resize - img1: {img1.shape if img1 is not None else None}, img2: {img2.shape if img2 is not None else None}")
                    return True  # Default to accepting if images are invalid
                
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                
                # Ensure valid dimensions
                if h1 <= 0 or w1 <= 0 or h2 <= 0 or w2 <= 0:
                    print(f"DEBUG: Invalid image dimensions - img1: {w1}x{h1}, img2: {w2}x{h2}")
                    return True  # Default to accepting if dimensions are invalid
                
                target_size = (min(w1, w2, 384), min(h1, h2, 384))  # Max 384x384 for better feature extraction (balance between speed and quality)
                
                try:
                    img1_resized = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
                    img2_resized = cv2.resize(img2, target_size, interpolation=cv2.INTER_AREA)
                except Exception as resize_error:
                    print(f"DEBUG: Image resize failed: {resize_error} (img1: {w1}x{h1}, img2: {w2}x{h2}, target: {target_size})")
                    return True  # Default to accepting if resize fails
                
                if method == 'orb':
                    print(f"DEBUG: Calling _verify_orb for {os.path.basename(file1)} <-> {os.path.basename(file2)}")
                    result = self._verify_orb(img1_resized, img2_resized, file1, file2)
                    print(f"DEBUG: _verify_orb returned: {result}")
                    return result
                elif method == 'sift':
                    return self._verify_sift(img1_resized, img2_resized)
                else:  # 'ssim' or default
                    return self._verify_ssim(img1_resized, img2_resized)
            finally:
                # Restore OpenCV log level
                try:
                    cv2.setLogLevel(original_log_level)
                except:
                    pass
            
        except Exception as e:
            print(f"DEBUG: OpenCV {method} verification failed: {e}")
            return True  # Default to accepting on error
    
    def _imread_safe(self, file_path: str, flags: int = None) -> Optional['np.ndarray']:
        """
        Safely read image file using OpenCV, handling Windows path encoding issues.
        
        On Windows, OpenCV's cv2.imread() may fail with non-ASCII characters in paths.
        This function tries multiple methods to read the file, prioritizing methods that
        don't trigger OpenCV warnings.
        
        Args:
            file_path: Path to image file
            flags: OpenCV imread flags (default: IMREAD_GRAYSCALE)
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            import cv2
            import numpy as np
            
            # Set default flags if not provided
            if flags is None:
                flags = cv2.IMREAD_GRAYSCALE
            
            # Check if file exists first
            if not os.path.exists(file_path):
                return None
            
            # Save and set log level at the start to suppress all warnings
            try:
                original_log_level = cv2.getLogLevel()
            except:
                original_log_level = 2  # Default to WARN if getLogLevel fails
            
            try:
                cv2.setLogLevel(3)  # Set to ERROR to suppress WARN messages
            except:
                pass  # If setLogLevel fails, continue anyway
            
            # Check if file is TIFF or DNG format - OpenCV cannot handle 10-bit TIFF and other special formats
            # DNG (Digital Negative) is based on TIFF, so OpenCV may try to read it as TIFF
            # For all TIFF/DNG files, use PIL or rawpy directly to avoid OpenCV warnings
            file_ext = os.path.splitext(file_path.lower())[1]
            is_tiff = file_ext in {'.tif', '.tiff'}
            is_dng = file_ext == '.dng'
            
            # For DNG files, check if it's a RAW file first (use rawpy if available)
            if is_dng and self.is_raw_image_file(file_path):
                # DNG is a RAW format, try rawpy first
                try:
                    import rawpy
                    with rawpy.imread(file_path) as raw:
                        rgb = raw.postprocess(use_camera_wb=True, half_size=False)
                        if len(rgb.shape) == 3:
                            gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
                        else:
                            gray = rgb.astype(np.uint8)
                        return gray
                except Exception:
                    # rawpy failed, fall through to PIL
                    pass
            
            # For ALL TIFF files (including DNG if rawpy failed), use PIL directly to avoid OpenCV warnings
            # PIL can handle all TIFF formats including 10-bit, 12-bit, 14-bit, 16-bit, etc.
            if is_tiff or is_dng:
                try:
                    from PIL import Image
                    pil_img = Image.open(file_path)
                    # Convert to grayscale if needed
                    if pil_img.mode != 'L':
                        pil_img = pil_img.convert('L')
                    # Convert PIL image to numpy array (uint8)
                    img_array = np.array(pil_img, dtype=np.uint8)
                    pil_img.close()  # Close the image to free resources
                    return img_array
                except Exception as tiff_error:
                    # PIL failed, try OpenCV as fallback (but this may trigger warnings)
                    print(f"DEBUG: PIL failed to read TIFF/DNG {os.path.basename(file_path)}, trying OpenCV: {tiff_error}")
                    pass
            
            # Skip OpenCV methods for TIFF/DNG files to avoid warnings
            # These formats should already be handled above with PIL/rawpy
            if not (is_tiff or is_dng):
                # Method 1: Use numpy and cv2.imdecode FIRST (avoids path encoding warnings)
                # This method reads the file as binary and decodes it, bypassing path encoding issues
                try:
                    with open(file_path, 'rb') as f:
                        img_data = np.frombuffer(f.read(), np.uint8)
                        img = cv2.imdecode(img_data, flags)
                        if img is not None:
                            return img
                except (IOError, OSError, UnicodeDecodeError, Exception):
                    # File read failed, try other methods
                    pass
                
                # Method 2: Try standard cv2.imread (works for simple paths)
                try:
                    img = cv2.imread(file_path, flags)
                    if img is not None:
                        return img
                except Exception:
                    pass
                
                # Method 3: Try with Windows long path prefix (Windows-specific)
                # This helps with paths containing special characters
                try:
                    import sys
                    if sys.platform == 'win32':
                        # Use Windows long path prefix to handle special characters
                        if not file_path.startswith('\\\\?\\'):
                            long_path = '\\\\?\\' + os.path.abspath(file_path)
                            img = cv2.imread(long_path, flags)
                            if img is not None:
                                return img
                except Exception:
                    pass
                
                return None
            
            # If we reach here and it's TIFF/DNG but PIL failed, return None
            return None
        except Exception:
            return None
        finally:
            # Restore original log level
            try:
                cv2.setLogLevel(original_log_level)
            except:
                pass
    
    def _load_image_alternative(self, file_path: str) -> Optional['np.ndarray']:
        """
        Load image using alternative methods when OpenCV fails.
        Handles: 10-bit TIFF, RAW formats (ARW, CR2, NEF, etc.)
        
        Args:
            file_path: Path to image file
            
        Returns:
            Grayscale numpy array or None if failed
        """
        try:
            import numpy as np
            import cv2
            
            # Suppress OpenCV warnings during alternative loading
            try:
                original_log_level = cv2.getLogLevel()
            except:
                original_log_level = 2
            
            try:
                cv2.setLogLevel(3)  # Set to ERROR to suppress WARN messages
            except:
                pass
            
            try:
                # Check if it's a RAW file
                if self.is_raw_image_file(file_path):
                    # Try rawpy for RAW files
                    try:
                        import rawpy
                        with rawpy.imread(file_path) as raw:
                            # Extract RGB image and convert to grayscale
                            rgb = raw.postprocess(use_camera_wb=True, half_size=False)
                            # Convert RGB to grayscale using standard weights
                            if len(rgb.shape) == 3:
                                gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
                            else:
                                gray = rgb.astype(np.uint8)
                            return gray
                    except Exception:
                        # rawpy failed, try PIL
                        pass
                
                # Try PIL for other formats (10-bit TIFF, etc.)
                try:
                    from PIL import Image
                    pil_img = Image.open(file_path)
                    # Convert to grayscale if needed
                    if pil_img.mode != 'L':
                        pil_img = pil_img.convert('L')
                    # Convert PIL image to numpy array
                    return np.array(pil_img, dtype=np.uint8)
                except Exception:
                    # Both methods failed
                    return None
            finally:
                # Restore OpenCV log level
                try:
                    cv2.setLogLevel(original_log_level)
                except:
                    pass
                
        except Exception:
            return None
    
    def _get_orb_descriptors(self, file_path: str, img: 'np.ndarray') -> tuple:
        """
        Get ORB descriptors for an image, using cache if available.
        Supports GPU acceleration if available and enabled.
        Includes image preprocessing to improve feature extraction for low-contrast images.
        
        Args:
            file_path: Path to the image file (for caching)
            img: Grayscale image (numpy array)
            
        Returns:
            Tuple of (keypoints, descriptors) or (None, None) on error
        """
        try:
            import cv2
            import numpy as np
            
            # Check cache first
            with self._orb_cache_lock:
                if file_path in self._orb_cache:
                    return self._orb_cache[file_path]
            
            # Preprocess image to improve feature extraction for low-contrast images
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
            processed_img = img.copy()
            img_min, img_max = img.min(), img.max()
            
            # Only apply enhancement if image has low contrast (range < 100)
            if img_max - img_min < 100 and img_max - img_min > 0:
                try:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    processed_img = clahe.apply(img)
                except Exception:
                    # If CLAHE fails, try simple histogram equalization
                    try:
                        processed_img = cv2.equalizeHist(img)
                    except Exception:
                        processed_img = img  # Fallback to original
            
            # Use GPU acceleration if available and enabled
            if self.use_orb_gpu:
                try:
                    # Upload image to GPU
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(processed_img)
                    
                    # Create ORB detector on GPU (note: OpenCV doesn't have direct GPU ORB,
                    # but we can use GPU for image processing and CPU for ORB)
                    # Optimized parameters for duplicate detection (2-3x faster)
                    orb = cv2.ORB_create(
                        nfeatures=150,       # Reduced from 300 (faster matching, sufficient for duplicates)
                        fastThreshold=25,    # Increased from 20 (fewer keypoints, faster detection)
                        scaleFactor=1.3,     # Larger scale factor (fewer pyramid levels)
                        nlevels=6,           # Fewer levels (less scale invariance, faster)
                        edgeThreshold=15     # Ignore edges (cleaner features)
                    )
                    
                    # Download back to CPU for ORB (ORB doesn't have GPU implementation in OpenCV)
                    # However, we can use GPU for resizing/processing if needed
                    cpu_img = gpu_img.download()
                    kp, des = orb.detectAndCompute(cpu_img, None)
                except Exception as gpu_error:
                    print(f"DEBUG: GPU ORB failed, falling back to CPU: {gpu_error}")
                    # Optimized parameters for duplicate detection
                    orb = cv2.ORB_create(
                        nfeatures=150,
                        fastThreshold=25,
                        scaleFactor=1.3,
                        nlevels=6,
                        edgeThreshold=15
                    )
                    kp, des = orb.detectAndCompute(processed_img, None)
            else:
                # Standard CPU implementation - optimized for duplicate detection (2-3x faster)
                orb = cv2.ORB_create(
                    nfeatures=150,       # Down from 300 (faster matching, still accurate for duplicates)
                    fastThreshold=25,    # Up from 20 (fewer keypoints)
                    scaleFactor=1.3,     # Larger than default 1.2 (fewer pyramid levels)
                    nlevels=6,           # Down from default 8 (faster)
                    edgeThreshold=15     # Filter edge keypoints (cleaner features)
                )
                kp, des = orb.detectAndCompute(processed_img, None)
            
            # Validate image before computing descriptors
            if img is None or img.size == 0:
                print(f"DEBUG: ORB descriptor computation failed: invalid image for {os.path.basename(file_path) if file_path else 'unknown'}")
                return (None, None)
            
            # Check if descriptors were computed successfully
            if des is None:
                print(f"DEBUG: ORB descriptor computation returned None for {os.path.basename(file_path) if file_path else 'unknown'} (image shape: {img.shape})")
                return (None, None)
            
            # Cache the result with size limiting
            if len(des) >= 10:
                with self._orb_cache_lock:
                    # Enforce cache size limit to prevent unbounded growth
                    if len(self._orb_cache) >= self._orb_cache_max_size:
                        # Remove oldest entry (FIFO eviction)
                        oldest_key = next(iter(self._orb_cache))
                        self._orb_cache.pop(oldest_key, None)
                    self._orb_cache[file_path] = (kp, des)
                return (kp, des)
            else:
                print(f"DEBUG: ORB descriptor computation: insufficient features ({len(des)} < 10) for {os.path.basename(file_path) if file_path else 'unknown'} (image shape: {img.shape})")
                return (None, None)
            
        except Exception as e:
            print(f"DEBUG: ORB descriptor computation error for {os.path.basename(file_path) if file_path else 'unknown'}: {e}")
            return (None, None)
    
    def _verify_orb(self, img1: 'np.ndarray', img2: 'np.ndarray', file1: str = None, file2: str = None) -> bool:
        """
        Verify similarity using ORB (Oriented FAST and Rotated BRIEF).
        Fastest method (~100x faster than SIFT), good accuracy.
        Uses cached descriptors if file paths are provided.
        Supports GPU-accelerated matching if available.
        
        Args:
            img1, img2: Grayscale images (numpy arrays)
            file1, file2: Optional file paths for caching
            
        Returns:
            True if images are similar (enough matching features)
        """
        try:
            import cv2
            import numpy as np
            
            # Get descriptors (from cache if available)
            if file1 and file2:
                kp1, des1 = self._get_orb_descriptors(file1, img1)
                kp2, des2 = self._get_orb_descriptors(file2, img2)
            else:
                # Fallback: compute without caching
                orb = cv2.ORB_create(nfeatures=300, fastThreshold=20)  # Reduced features for speed
                kp1, des1 = orb.detectAndCompute(img1, None)
                kp2, des2 = orb.detectAndCompute(img2, None)
            
            # Check if images were successfully read and have valid data
            if img1 is None or img2 is None or (hasattr(img1, 'size') and img1.size == 0) or (hasattr(img2, 'size') and img2.size == 0):
                print(f"DEBUG: ORB verification failed: invalid images (img1: {img1.shape if img1 is not None and hasattr(img1, 'shape') else None}, img2: {img2.shape if img2 is not None and hasattr(img2, 'shape') else None})")
                return False
            
            # Check if keypoints and descriptors are valid
            if kp1 is None or kp2 is None or des1 is None or des2 is None:
                # Not enough features for ORB matching
                # Use fallback method: histogram comparison for low-contrast/solid color images
                img1_info = f"shape={img1.shape}, dtype={img1.dtype}, min={img1.min()}, max={img1.max()}" if img1 is not None and hasattr(img1, 'shape') else "None"
                img2_info = f"shape={img2.shape}, dtype={img2.dtype}, min={img2.min()}, max={img2.max()}" if img2 is not None and hasattr(img2, 'shape') else "None"
                print(f"DEBUG: ORB verification failed: invalid keypoints/descriptors (kp1: {kp1 is not None}, kp2: {kp2 is not None}, des1: {des1 is not None}, des2: {des2 is not None})")
                if file1 and file2:
                    print(f"DEBUG: Files - file1: {os.path.basename(file1)}, file2: {os.path.basename(file2)}")
                print(f"DEBUG: Image info - img1: {img1_info}, img2: {img2_info}")
                
                # Fallback: Use histogram comparison for images that can't extract ORB features
                # This helps with low-contrast, solid color, or very similar images
                try:
                    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
                    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    # If histograms are very similar (correlation > 0.95), consider them duplicates
                    if correlation > 0.95:
                        print(f"DEBUG: ORB fallback: histogram correlation={correlation:.3f} > 0.95, returning True")
                        return True
                    else:
                        print(f"DEBUG: ORB fallback: histogram correlation={correlation:.3f} <= 0.95, returning False")
                        return False
                except Exception as hist_error:
                    print(f"DEBUG: ORB fallback histogram comparison failed: {hist_error}")
                    # If histogram comparison also fails, default to False (not similar)
                    return False
            
            if len(des1) < 10 or len(des2) < 10:
                # Not enough features for ORB matching
                # Use fallback method: histogram comparison for low-contrast/solid color images
                img1_info = f"shape={img1.shape}, dtype={img1.dtype}, min={img1.min()}, max={img1.max()}" if img1 is not None and hasattr(img1, 'shape') else "None"
                img2_info = f"shape={img2.shape}, dtype={img2.dtype}, min={img2.min()}, max={img2.max()}" if img2 is not None and hasattr(img2, 'shape') else "None"
                print(f"DEBUG: ORB verification failed: insufficient features (des1: {len(des1) if des1 is not None else 0}, des2: {len(des2) if des2 is not None else 0})")
                if file1 and file2:
                    print(f"DEBUG: Files - file1: {os.path.basename(file1)}, file2: {os.path.basename(file2)}")
                print(f"DEBUG: Image info - img1: {img1_info}, img2: {img2_info}")
                
                # Fallback: Use histogram comparison for images that can't extract ORB features
                # This helps with low-contrast, solid color, or very similar images
                try:
                    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
                    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    # If histograms are very similar (correlation > 0.95), consider them duplicates
                    if correlation > 0.95:
                        print(f"DEBUG: ORB fallback: histogram correlation={correlation:.3f} > 0.95, returning True")
                        return True
                    else:
                        print(f"DEBUG: ORB fallback: histogram correlation={correlation:.3f} <= 0.95, returning False")
                        return False
                except Exception as hist_error:
                    print(f"DEBUG: ORB fallback histogram comparison failed: {hist_error}")
                    # If histogram comparison also fails, default to False (not similar)
                    return False
            
            # Optimized matching: Use BFMatcher with early termination
            # For ORB, BFMatcher with Hamming distance is faster than FLANN for binary descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test (Lowe's ratio test) with early termination
            good_matches = []
            # Use descriptor count instead of keypoint count (more reliable)
            min_features = min(len(des1), len(des2))
            required_matches = int(min_features * 0.12)  # Need at least 12% match ratio (reduced from 0.15)
            
            # Early termination: if we can't possibly get enough matches, return early
            if len(matches) < required_matches:
                return False
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Ratio threshold
                        good_matches.append(m)
                        # Early termination: if we already have enough matches, stop checking
                        if len(good_matches) >= required_matches:
                            break
            
            # Calculate match ratio using descriptor count (more reliable than keypoint count)
            match_ratio = len(good_matches) / min_features
            
            # Threshold: at least 12% of features should match for duplicates (reduced from 0.15 to 0.12 to reduce false negatives)
            result = match_ratio >= 0.12
            # Use descriptor count for display (more reliable than keypoint count)
            kp1_count = len(kp1) if kp1 is not None else len(des1)
            kp2_count = len(kp2) if kp2 is not None else len(des2)
            print(f"DEBUG: ORB match details: {len(good_matches)} good matches out of {kp1_count}/{kp2_count} features, ratio={match_ratio:.3f}, threshold=0.12, result={result}")
            return result
            
        except Exception as e:
            print(f"DEBUG: ORB verification error: {e}")
            return True
    
    def _verify_sift(self, img1: 'np.ndarray', img2: 'np.ndarray') -> bool:
        """
        Verify similarity using SIFT (Scale-Invariant Feature Transform).
        Slowest but most accurate method.
        
        Args:
            img1, img2: Grayscale images (numpy arrays)
            
        Returns:
            True if images are similar (enough matching features)
        """
        try:
            import cv2
            import numpy as np
            
            # Initialize SIFT detector
            sift = cv2.SIFT_create(nfeatures=500)  # Limit features for speed
            
            # Find keypoints and descriptors
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            
            # Check if keypoints and descriptors are valid
            if kp1 is None or kp2 is None or des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                # Not enough features, likely not similar
                return False
            
            # Match features using FLANN matcher (faster than BF for SIFT)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test (Lowe's ratio test)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Ratio threshold
                        good_matches.append(m)
            
            # Calculate match ratio using descriptor count (more reliable than keypoint count)
            min_features = min(len(des1), len(des2))
            match_ratio = len(good_matches) / min_features
            
            # Threshold: at least 12% of features should match for duplicates (reduced from 0.15 to 0.12 for consistency)
            return match_ratio >= 0.12
            
        except Exception as e:
            print(f"DEBUG: SIFT verification error: {e}")
            return True
    
    def _verify_ssim(self, img1: 'np.ndarray', img2: 'np.ndarray') -> bool:
        """
        Verify similarity using SSIM (Structural Similarity Index).
        Medium speed, high accuracy.
        
        Args:
            img1, img2: Grayscale images (numpy arrays)
            
        Returns:
            True if images are similar (SSIM > 0.85)
        """
        try:
            import cv2
            import numpy as np
            
            # Using a simplified SSIM calculation (mean squared error normalized)
            # Full SSIM is more complex but this approximation is faster
            diff = cv2.absdiff(img1, img2)
            mse = np.mean(diff ** 2)
            
            # Normalize MSE to 0-1 scale (assuming 8-bit images, max value 255)
            normalized_mse = mse / (255.0 ** 2)
            
            # Convert MSE to similarity score (lower MSE = higher similarity)
            # SSIM-like score: 1.0 = identical, 0.0 = completely different
            similarity = 1.0 - min(1.0, normalized_mse)
            
            # Threshold: 0.85 means images are 85% similar
            return similarity >= 0.85
            
        except Exception as e:
            print(f"DEBUG: SSIM verification error: {e}")
            return True
    
    def verify_group_before_deletion(self, group_files: List[str]) -> bool:
        """
        Perform on-demand ORB verification for a group of potential duplicates.
        Call this before deleting files when defer_orb_verification is enabled.
        
        For large datasets (10K+ images), ORB verification is deferred to deletion time
        to provide 5-10x speedup during scan. This method verifies the group on-demand.
        
        Args:
            group_files: List of file paths in the duplicate group
            
        Returns:
            True if group is verified as duplicates (safe to delete selections)
            False if verification fails (group may contain false positives)
        """
        # If ORB verification is disabled or not deferred, always return True
        if not self.opencv_verification_method or not self.defer_orb_verification:
            return True
        
        # Need at least 2 files to verify
        if len(group_files) < 2:
            return True
        
        try:
            # Verify first file against second as representative check
            # This is much faster than checking all pairs and sufficient for most cases
            # If the hash matched them, they're likely duplicates; ORB just reduces false positives
            result = self._verify_similarity_opencv(
                "", "",  # Hash strings not needed for direct file comparison
                [group_files[0]], [group_files[1]], 
                method=self.opencv_verification_method
            )
            
            if result:
                print(f"DEBUG: On-demand ORB verification PASSED for group: {os.path.basename(group_files[0])} <-> {os.path.basename(group_files[1])}")
            else:
                print(f"DEBUG: On-demand ORB verification FAILED for group: {os.path.basename(group_files[0])} <-> {os.path.basename(group_files[1])}")
                print(f"DEBUG: This group may contain false positives. Review manually before deletion.")
            
            return result
        except Exception as e:
            print(f"DEBUG: On-demand verification error: {e}")
            # On error, default to accepting (hash matching is already reliable)
            return True
    
    
    def _prefilter_exact_duplicates(self, files: List[str]) -> tuple:
        """
        Pre-filter files to identify exact duplicates by size and name+timestamp.
        This reduces the number of files that need hash calculation.
        
        Args:
            files: List of file paths to check
            
        Returns:
            tuple: (exact_duplicate_groups, remaining_files)
                - exact_duplicate_groups: Dict mapping group_id to list of duplicate files
                - remaining_files: List of files that still need hash calculation
        """
        if len(files) < 2:
            return {}, files
        
        exact_duplicate_groups = {}
        remaining_files = []
        
        # Group files by size first
        size_groups = defaultdict(list)
        file_info = {}  # path -> (size, mtime, basename)
        
        for file_path in files:
            try:
                st = os.stat(file_path)
                size = st.st_size
                mtime = st.st_mtime
                basename = os.path.splitext(os.path.basename(file_path))[0].lower()
                size_groups[size].append(file_path)
                file_info[file_path] = (size, mtime, basename)
            except (OSError, PermissionError):
                remaining_files.append(file_path)
                continue
        
        # Check each size group for exact duplicates
        processed_files = set()
        group_id_counter = 0
        
        for size, size_group_files in size_groups.items():
            if len(size_group_files) < 2:
                remaining_files.extend(size_group_files)
                continue
            
            # Strategy 1: Same size + same name + same timestamp = exact duplicate
            # Group by (basename, timestamp_bucket)
            name_timestamp_groups = defaultdict(list)
            for file_path in size_group_files:
                if file_path in file_info:
                    size_val, mtime, basename = file_info[file_path]
                    # Round timestamp to nearest second for matching
                    timestamp_bucket = int(mtime)
                    name_timestamp_groups[(basename, timestamp_bucket)].append(file_path)
            
            # If all files in this size group share the same (basename, timestamp), they're exact duplicates
            for (basename, ts), group_files in name_timestamp_groups.items():
                if len(group_files) >= 2:
                    # All files with same size, name, and timestamp = exact duplicates
                    group_id = f"exact_size_name_ts_{group_id_counter}"
                    exact_duplicate_groups[group_id] = group_files
                    processed_files.update(group_files)
                    group_id_counter += 1
                    print(f"DEBUG: Pre-filter exact duplicate (size+name+timestamp): {len(group_files)} files, name={basename}, size={size}")
            
            # Strategy 2: Same size + exactly 2 files = likely exact duplicates (for non-image files)
            # For image files, we still need perceptual hash as same size doesn't mean same image
            # But for non-image files (like documents), same size often means exact duplicate
            for file_path in size_group_files:
                if file_path not in processed_files:
                    # Check if this file is part of a same-size pair
                    same_size_files = [f for f in size_group_files if f not in processed_files]
                    if len(same_size_files) == 2:
                        # Only 2 files with same size = likely exact duplicates
                        # But we need to be careful: for images, same size doesn't mean same content
                        # So we only apply this for non-image files
                        if not self.is_image_file(same_size_files[0]):
                            group_id = f"exact_size_pair_{group_id_counter}"
                            exact_duplicate_groups[group_id] = same_size_files
                            processed_files.update(same_size_files)
                            group_id_counter += 1
                            print(f"DEBUG: Pre-filter exact duplicate (same size pair, non-image): {len(same_size_files)} files, size={size}")
                        else:
                            # For images, we still need hash calculation
                            remaining_files.extend(same_size_files)
                            processed_files.update(same_size_files)
                    else:
                        # More than 2 files with same size, need hash calculation
                        remaining_files.append(file_path)
        
        # Add any unprocessed files
        for file_path in files:
            if file_path not in processed_files:
                remaining_files.append(file_path)
        
        return exact_duplicate_groups, remaining_files
    
    def _merge_raw_jpeg_by_timestamp(self, duplicate_groups: Dict[str, List[str]], time_threshold: float = 5.0) -> Dict[str, List[str]]:
        """
        Merge RAW and JPEG files that have similar timestamps into the same group.
        
        This helps identify the same photo in different formats (RAW and JPEG) even if
        they have different perceptual hashes due to processing differences.
        
        Args:
            duplicate_groups: Dictionary mapping hash values to lists of file paths
            time_threshold: Maximum time difference in seconds to consider files as related (default: 5.0)
        
        Returns:
            Updated duplicate_groups dictionary with RAW/JPEG pairs merged
        """
        if not duplicate_groups:
            return duplicate_groups
        
        # Collect all files with their timestamps and types
        file_info = {}  # path -> (mtime, is_raw, hash_value)
        
        for hash_val, files in duplicate_groups.items():
            for file_path in files:
                try:
                    st = os.stat(file_path)
                    mtime = st.st_mtime
                    is_raw = self.is_raw_image_file(file_path)
                    file_info[file_path] = (mtime, is_raw, hash_val)
                except (OSError, PermissionError):
                    continue
        
        # Find RAW/JPEG pairs with similar timestamps
        merged_groups = duplicate_groups.copy()
        processed_pairs = set()
        
        # Group files by base name (without extension) and similar timestamp
        base_name_groups = defaultdict(list)
        for file_path, (mtime, is_raw, hash_val) in file_info.items():
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            # Normalize base name (remove common suffixes like _1, -copy, etc.)
            normalized_base = base_name.lower().rsplit('_', 1)[0].rsplit('-', 1)[0]
            time_bucket = int(mtime / time_threshold)
            base_name_groups[(normalized_base, time_bucket)].append((file_path, mtime, is_raw, hash_val))
        
        # Check for RAW/JPEG pairs in the same base name group
        for (base_name, time_bucket), files in base_name_groups.items():
            if len(files) < 2:
                continue
            
            # Separate RAW and JPEG files
            raw_files = [(fp, mt, h) for fp, mt, is_raw, h in files if is_raw]
            jpeg_files = [(fp, mt, h) for fp, mt, is_raw, h in files if not is_raw and self.is_image_file(fp)]
            
            # Try to match RAW and JPEG files with similar timestamps
            for raw_path, raw_mtime, raw_hash in raw_files:
                for jpeg_path, jpeg_mtime, jpeg_hash in jpeg_files:
                    # Skip if already in the same group
                    if raw_hash == jpeg_hash:
                        continue
                    
                    # Check timestamp difference
                    time_diff = abs(raw_mtime - jpeg_mtime)
                    if time_diff <= time_threshold:
                        # Merge the two groups
                        pair_key = tuple(sorted([raw_hash, jpeg_hash]))
                        if pair_key in processed_pairs:
                            continue
                        processed_pairs.add(pair_key)
                        
                        # Use the smaller hash as the key (for consistency)
                        target_hash = min(raw_hash, jpeg_hash)
                        source_hash = max(raw_hash, jpeg_hash)
                        
                        if target_hash in merged_groups and source_hash in merged_groups:
                            # Merge source group into target group
                            target_files = set(merged_groups[target_hash])
                            source_files = set(merged_groups[source_hash])
                            merged_groups[target_hash] = list(target_files | source_files)
                            # Remove the source group
                            del merged_groups[source_hash]
                            print(f"DEBUG: Merged RAW/JPEG pair by timestamp: {os.path.basename(raw_path)} and {os.path.basename(jpeg_path)} (time diff: {time_diff:.2f}s)")
        
        return merged_groups
    
    def scan_duplicate_files(self, scan_paths: List[str], use_imagehash: bool = False, use_multi_hash: bool = True, use_orb_verification: bool = False):
        """
        Main scanning function.
        
        Args:
            scan_paths: List of directory paths to scan
            use_imagehash: Whether to use perceptual hashing for images and videos
            use_multi_hash: If True, use multi-algorithm hash; if False, use single algorithm (phash)
            use_orb_verification: If True, use ORB verification for perceptual hash matches
        """
        try:
            self.scan_cancelled = False
            self.use_imagehash = use_imagehash and IMAGEHASH_AVAILABLE
            self.use_multi_hash = use_multi_hash if self.use_imagehash else False
            
            # Set OpenCV verification method if ORB verification is enabled
            if use_orb_verification:
                self.opencv_verification_method = 'orb'
            else:
                self.opencv_verification_method = None
            
            # Debug: Log hash mode and availability
            print(f"DEBUG: scan_duplicate_files called with use_imagehash={use_imagehash}, use_multi_hash={use_multi_hash}, use_orb_verification={use_orb_verification}")
            print(f"DEBUG: Final settings: use_imagehash={self.use_imagehash}, use_multi_hash={self.use_multi_hash}, opencv_method={self.opencv_verification_method}")
            
            # Check OpenCV availability (already imported at module level)
            if OPENCV_AVAILABLE:
                try:
                    import cv2
                    print(f"DEBUG: OpenCV available: {OPENCV_AVAILABLE}")
                    print(f"DEBUG: OpenCV version: {cv2.__version__}")
                except:
                    pass
            
            self.file_groups = defaultdict(list)
            self.files_scanned = 0
            self.total_files = 0
            
            # Reset cache stats
            with self._phash_stats_lock:
                self._phash_cache_lookups = 0
                self._phash_cache_hits = 0
                self._phash_cache_misses = 0
                self._phash_cache_puts = 0
                self._md5_cache_lookups = 0
                self._md5_cache_hits = 0
                self._md5_cache_misses = 0
                self._md5_cache_puts = 0
            
            if self.use_imagehash and self.phash_cache_enabled:
                self._ensure_phash_db()
            
            # Phase 1: Collect files by size
            size_groups = defaultdict(list)
            size_groups_lock = threading.Lock()
            total_collected = [0]
            total_collected_lock = threading.Lock()
            total_image_files = [0]
            total_image_files_lock = threading.Lock()
            all_image_files = []
            all_image_files_lock = threading.Lock()
            
            print(f"DEBUG: Engine scan_duplicate_files called with paths={scan_paths}")
            print(f"DEBUG: use_imagehash={self.use_imagehash}")
            
            # Phase 1: Collect files
            # Calculate total phases: base phases + ORB verification phase if enabled
            base_phases = 4 if not self.use_imagehash else 3  # Phase 3 skipped if using imagehash
            total_phases = base_phases + (1 if self.opencv_verification_method else 0)
            phase_num = 1
            # Reset progress bar to 0% for Phase 1 (each phase has its own progress bar)
            self.progress_callback(0.0)
            if self.use_imagehash:
                print("DEBUG: Calling status_callback: Collecting files (ImageHash enabled)...")
                self.status_callback(f"{phase_num}/{total_phases} Scanning: Collecting files...")
            else:
                print("DEBUG: Calling status_callback: Collecting files...")
                self.status_callback(f"{phase_num}/{total_phases} Scanning: Collecting files...")
            
            scan_tasks = list(scan_paths)
            if len(scan_tasks) == 1:
                try:
                    root_path = scan_tasks[0]
                    subdirs = []
                    try:
                        with os.scandir(root_path) as it:
                            for entry in it:
                                try:
                                    if entry.is_dir(follow_symlinks=False):
                                        if entry.name.lower() not in self._get_system_dirs():
                                            subdirs.append(entry.path)
                                    elif entry.is_file(follow_symlinks=False):
                                        if entry.name.startswith('._'):
                                            continue
                                        
                                        # Only process supported file types (images, videos, audio, PDFs, EPUBs)
                                        if not self.is_supported_file(entry.name):
                                            continue
                                        
                                        st = entry.stat(follow_symlinks=False)
                                        if st.st_size > 0:
                                            size_groups[st.st_size].append(entry.path)
                                            total_collected[0] += 1
                                            if self.is_image_file(entry.name):
                                                total_image_files[0] += 1
                                                if self.use_imagehash:
                                                    with all_image_files_lock:
                                                        all_image_files.append(entry.path)
                                except (OSError, PermissionError):
                                    continue
                    except Exception:
                        pass
                    
                    if subdirs:
                        scan_tasks = subdirs
                except Exception:
                    scan_tasks = scan_paths
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for path in scan_tasks:
                    if self.scan_cancelled:
                        return
                    future = executor.submit(
                        self.collect_files_by_size,
                        path, size_groups, total_collected, size_groups_lock, total_collected_lock,
                        total_image_files, total_image_files_lock,
                        all_image_files if self.use_imagehash else None,
                        all_image_files_lock if self.use_imagehash else None
                    )
                    futures.append(future)
                
                last_status_update = 0
                last_update_time = time.time()
                min_update_interval = 0.2  # 200ms minimum between collection status updates
                
                for future in as_completed(futures):
                    if self.scan_cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        return
                    try:
                        future.result()
                        # Update status periodically during collection with throttling
                        current_time = time.time()
                        with total_collected_lock:
                            count = total_collected[0]
                        
                        # Update if: (1) 1000+ files since last update, OR (2) 200ms+ since last update
                        if (count - last_status_update >= 1000) or (current_time - last_update_time >= min_update_interval):
                            # Estimate total files for progress calculation (use a reasonable estimate)
                            # For Phase 1, we don't know total yet, so use a conservative estimate
                            # Each phase has its own progress bar (0-100% per phase)
                            estimated_total = max(count * 2, 10000)  # Conservative estimate
                            phase_progress = min(count / estimated_total, 1.0) if estimated_total > 0 else 0.0
                            # Pass phase progress (0.0-1.0) instead of overall progress
                            self.progress_callback(phase_progress)
                            self.status_callback(f"{phase_num}/{total_phases} Scanning: Collected {count:,} files...")
                            last_status_update = count
                            last_update_time = current_time
                    except Exception as e:
                        print(f"Scan task error: {e}")
                        continue
                
                # Final status update for Phase 1
                with total_collected_lock:
                    final_count = total_collected[0]
                with total_image_files_lock:
                    img_count = total_image_files[0]
                # Phase 1 complete: set progress to 100% (1.0) for this phase
                self.progress_callback(1.0)
                if img_count > 0:
                    self.status_callback(f"{phase_num}/{total_phases} Scanning: Collected {final_count:,} files ({img_count:,} images)")
                else:
                    self.status_callback(f"{phase_num}/{total_phases} Scanning: Collected {final_count:,} files")
            
            # Phase 2: Filter potential duplicates
            phase_num = 2
            print(f"DEBUG: Phase 2 - Filtering potential duplicates")
            # Reset progress bar to 0% for Phase 2 (each phase has its own progress bar)
            self.progress_callback(0.0)
            self.status_callback(f"{phase_num}/{total_phases} Scanning: Filtering potential duplicates...")
            potential_duplicates_by_size = []
            
            # Pre-filter: Identify exact duplicates by size and name+timestamp (skip hash calculation)
            # These will be added directly to results without hash calculation
            exact_duplicate_groups = {}  # hash_val -> list of files
            exact_duplicate_count = 0
            
            if self.use_imagehash:
                image_files_for_hash = list(dict.fromkeys(all_image_files))
                non_image_files_by_size = []
                
                for size, files in size_groups.items():
                    if len(files) > 1:
                        non_image_in_group = [fp for fp in files if not self.is_image_file(fp)]
                        if len(non_image_in_group) > 1:
                            # Pre-filter: Check for exact duplicates by size and name+timestamp
                            exact_groups, remaining_files = self._prefilter_exact_duplicates(non_image_in_group)
                            exact_duplicate_groups.update(exact_groups)
                            exact_duplicate_count += sum(len(files) for files in exact_groups.values())
                            non_image_files_by_size.extend(remaining_files)
                
                # Pre-filter image files by size and name+timestamp
                if len(image_files_for_hash) > 1:
                    exact_image_groups, remaining_image_files = self._prefilter_exact_duplicates(image_files_for_hash)
                    exact_duplicate_groups.update(exact_image_groups)
                    exact_duplicate_count += sum(len(files) for files in exact_image_groups.values())
                    potential_duplicates_by_size.extend(remaining_image_files)
                else:
                    potential_duplicates_by_size.extend(image_files_for_hash)
                
                potential_duplicates_by_size.extend(non_image_files_by_size)
            else:
                for size, files in size_groups.items():
                    if len(files) > 1:
                        # Pre-filter: Check for exact duplicates by size and name+timestamp
                        exact_groups, remaining_files = self._prefilter_exact_duplicates(files)
                        exact_duplicate_groups.update(exact_groups)
                        exact_duplicate_count += sum(len(files) for files in exact_groups.values())
                        potential_duplicates_by_size.extend(remaining_files)
            
            if exact_duplicate_count > 0:
                print(f"DEBUG: Pre-filtered {exact_duplicate_count} files as exact duplicates (skipping hash calculation)")
                print(f"DEBUG: Found {len(exact_duplicate_groups)} exact duplicate groups by size/name+timestamp")
            
            print(f"DEBUG: Found {len(potential_duplicates_by_size)} potential duplicate files (need hash calculation)")
            # Phase 2 complete: set progress to 100% (1.0) for this phase
            self.progress_callback(1.0)
            if not potential_duplicates_by_size:
                self.status_callback("No duplicate files found.")
                self.results_callback({})
                return
            
            # Phase 3: Partial hash pre-filtering (if not using imagehash)
            files_to_full_hash = []
            
            if self.use_imagehash:
                image_files = []
                non_image_files = []
                for fp in potential_duplicates_by_size:
                    if self.is_image_file(fp):
                        image_files.append(fp)
                    else:
                        non_image_files.append(fp)
                
                files_to_full_hash.extend(image_files)
                
                if non_image_files:
                    # Phase 2 for imagehash mode (3 phases total)
                    phase_num = 2
                    # Reset progress bar to 0% for partial hashing phase
                    self.progress_callback(0.0)
                    self.status_callback(f"{phase_num}/{total_phases} Scanning: Pre-filtering (Partial Hash) {len(non_image_files):,} files...")
                    partial_groups = self.perform_partial_hashing(non_image_files, phase_num=phase_num, total_phases=total_phases)
                    for files in partial_groups.values():
                        if len(files) > 1:
                            files_to_full_hash.extend(files)
            else:
                # Phase 3 for non-imagehash mode (4 phases total)
                phase_num = 3
                # Reset progress bar to 0% for partial hashing phase
                self.progress_callback(0.0)
                self.status_callback(f"{phase_num}/{total_phases} Scanning: Pre-filtering (Partial Hash) {len(potential_duplicates_by_size):,} files...")
                partial_groups = self.perform_partial_hashing(potential_duplicates_by_size, phase_num=phase_num, total_phases=total_phases)
                for files in partial_groups.values():
                    if len(files) > 1:
                        files_to_full_hash.extend(files)
            
            files_to_full_hash = list(dict.fromkeys(files_to_full_hash))
            self.total_files = len(files_to_full_hash)
            print(f"DEBUG: {self.total_files} files to hash")
            
            if self.total_files == 0:
                self.status_callback("No files to hash.")
                self.results_callback({})
                return
            
            # Phase 4: Full hash calculation (or Phase 3 if using imagehash)
            phase_num = 4 if not self.use_imagehash else 3
            is_3_phase = self.use_imagehash  # True if 3 phases, False if 4 phases
            # Reset progress bar to 0% for hash calculation phase
            self.progress_callback(0.0)
            hash_groups = defaultdict(list)
            processed = 0
            
            # Optimized progress update interval:
            # - For small batches (< 100 files): update every file (real-time)
            # - For medium batches (100-1000 files): update every 10 files
            # - For large batches (> 1000 files): update every 1% or every 50 files, whichever is more frequent
            # This balances real-time feedback with performance
            if self.total_files < 100:
                progress_update_interval = 1  # Every file for small batches
            elif self.total_files < 1000:
                progress_update_interval = 10  # Every 10 files for medium batches
            else:
                # For large batches, update every 1% or every 50 files (whichever is smaller)
                progress_update_interval = min(max(1, self.total_files // 100), 50)
            
            # Time-based throttling: update at most once per 100ms to avoid UI overload
            last_update_time = [time.time()]
            min_update_interval = 0.1  # 100ms minimum between updates
            
            # CPU processing - use more workers for I/O-intensive hash calculation
            if files_to_full_hash:
                # Batch I/O optimization: Pre-fetch next batch of files while processing current batch
                # This overlaps I/O with computation, reducing total processing time
                prefetch_batch_size = min(50, len(files_to_full_hash))
                if len(files_to_full_hash) > prefetch_batch_size:
                    # Prefetch next batch in background
                    next_batch = files_to_full_hash[prefetch_batch_size:prefetch_batch_size * 2]
                    self._prefetch_files_batch(next_batch, max_workers=min(8, self.hash_workers))
                
                # Use hash_workers (more threads) for I/O-bound hash calculation
                with ThreadPoolExecutor(max_workers=self.hash_workers) as executor:
                    future_to_file = {
                        executor.submit(self.calculate_file_hash, file_path): file_path
                        for file_path in files_to_full_hash
                    }
                    
                    # Prefetch next batch while processing
                    processed_count = 0
                    prefetch_trigger = prefetch_batch_size
                    
                    for future in as_completed(future_to_file):
                        if self.scan_cancelled:
                            executor.shutdown(wait=False, cancel_futures=True)
                            return
                        
                        file_path = future_to_file[future]
                        try:
                            file_hash = future.result()
                            if file_hash:
                                with self.hash_lock:
                                    if file_path not in hash_groups[file_hash]:
                                        hash_groups[file_hash].append(file_path)
                        except Exception as e:
                            print(f"Hash calculation error {file_path}: {e}")
                            continue
                        
                        processed += 1
                        processed_count += 1
                        self.files_scanned = processed
                        
                        # Trigger prefetch of next batch when we've processed enough files
                        if self._file_prefetch_enabled and processed_count >= prefetch_trigger and processed < len(files_to_full_hash):
                            # Prefetch next batch in background
                            next_start = processed
                            next_end = min(next_start + prefetch_batch_size, len(files_to_full_hash))
                            if next_start < len(files_to_full_hash):
                                next_batch = files_to_full_hash[next_start:next_end]
                                self._prefetch_files_batch(next_batch, max_workers=min(8, self.hash_workers))
                            prefetch_trigger += prefetch_batch_size
                        
                        # Update progress with throttling
                        current_time = time.time()
                        should_update = (
                            processed % progress_update_interval == 0 or 
                            processed == self.total_files or
                            (current_time - last_update_time[0]) >= min_update_interval
                        )
                        
                        if should_update:
                            # Calculate phase progress (0.0-1.0) for current phase only
                            phase_progress = processed / self.total_files if self.total_files > 0 else 0.0
                            # Pass phase progress (0.0-1.0) instead of overall progress
                            self.progress_callback(phase_progress)
                            self.status_callback(f"{phase_num}/{total_phases} Scanning: Calculating hash {processed:,}/{self.total_files:,}")
                            last_update_time[0] = current_time
            
            # Filter duplicate groups (without ORB verification first)
            # For multi-algorithm hash, use hamming distance comparison instead of exact match
            duplicate_groups = {}
            
            if self.use_multi_hash:
                # Multi-algorithm hash: Use hamming distance with voting mechanism
                # This reduces false positives by requiring multiple algorithms to agree
                self.status_callback(f"{phase_num}/{total_phases} Scanning: Grouping duplicates by similarity...")
                duplicate_groups = self._group_by_similarity_multi_hash(hash_groups, skip_orb_verification=True)
            else:
                # Single hash or MD5: Use exact match (original behavior)
                # ORB verification will be done in a separate phase
                self.status_callback(f"{phase_num}/{total_phases} Scanning: Grouping duplicates...")
                
                for hash_val, files in hash_groups.items():
                    if len(files) > 1:
                        unique_files = list(dict.fromkeys(files))
                        if len(unique_files) > 1:
                            # No ORB verification here - it will be done in a separate phase
                            duplicate_groups[hash_val] = unique_files
            
            # Phase 5 (or 4): ORB Verification (if enabled)
            if self.opencv_verification_method:
                phase_num = total_phases  # ORB verification is the last phase
                self.progress_callback(0.0)
                self.status_callback(f"{phase_num}/{total_phases} Scanning: Verifying duplicates with {self.opencv_verification_method.upper()}...")
                print(f"DEBUG: Starting ORB verification phase for {len(duplicate_groups)} groups")
                
                # Pre-filter: Separate files that can skip ORB verification
                # IMPORTANT: Even if files are pre-filtered as duplicates (same size/name+timestamp),
                # we still need to verify them against files with different sizes, because:
                # - File A and B may be pre-filtered as duplicates (same size)
                # - But file C (different size) might also be a duplicate of A/B
                # - So we need to verify A-C and B-C even though A-B is pre-filtered
                
                # Strategy: Within each hash_val group, separate files into:
                # - skip_orb_subgroups: Files that can skip ORB verification among themselves (same size/name+timestamp)
                # - need_orb_files: Files that need ORB verification
                # Then generate verification pairs:
                # 1. Within need_orb_files (normal verification)
                # 2. Between skip_orb_subgroups and need_orb_files (cross-verification)
                
                skip_orb_subgroups = {}  # hash_val -> list of subgroups (each subgroup can skip internal ORB)
                need_orb_files_by_hash = {}  # hash_val -> list of files that need ORB verification
                
                for hash_val, files in duplicate_groups.items():
                    unique_files = list(dict.fromkeys(files))
                    if len(unique_files) < 2:
                        continue
                    
                    # Check file properties
                    file_sizes = {}
                    file_timestamps = {}
                    file_basenames = {}
                    
                    for file_path in unique_files:
                        try:
                            st = os.stat(file_path)
                            file_sizes[file_path] = st.st_size
                            file_timestamps[file_path] = st.st_mtime
                            file_basenames[file_path] = os.path.splitext(os.path.basename(file_path))[0].lower()
                        except (OSError, PermissionError):
                            continue
                    
                    # Group files by size
                    size_groups = defaultdict(list)
                    for file_path in unique_files:
                        if file_path in file_sizes:
                            size_groups[file_sizes[file_path]].append(file_path)
                    
                    # Separate files into skip_orb_subgroups and need_orb_files
                    skip_subgroups = []
                    need_orb_files = []
                    
                    for size, size_files in size_groups.items():
                        if len(size_files) == 1:
                            # Single file with this size - needs ORB verification
                            need_orb_files.extend(size_files)
                        else:
                            # Multiple files with same size - check if they also have same name+timestamp
                            # Group by (basename, timestamp_bucket)
                            name_timestamp_groups = defaultdict(list)
                            for file_path in size_files:
                                if file_path in file_basenames and file_path in file_timestamps:
                                    basename = file_basenames[file_path]
                                    timestamp_bucket = int(file_timestamps[file_path])
                                    name_timestamp_groups[(basename, timestamp_bucket)].append(file_path)
                            
                            # If all files in this size group share the same (basename, timestamp), skip ORB
                            if len(name_timestamp_groups) == 1:
                                skip_subgroups.append(size_files)
                                basename, ts = list(name_timestamp_groups.keys())[0]
                                print(f"DEBUG: Skipping ORB for subgroup with same size+name+timestamp: {len(size_files)} files, size={size}, name={basename}, timestamp={ts}")
                            else:
                                # Files with same size but different name/timestamp - need ORB verification
                                need_orb_files.extend(size_files)
                    
                    if skip_subgroups:
                        skip_orb_subgroups[hash_val] = skip_subgroups
                    if need_orb_files:
                        need_orb_files_by_hash[hash_val] = need_orb_files
                
                total_skip_subgroups = sum(len(subs) for subs in skip_orb_subgroups.values())
                total_need_files = sum(len(files) for files in need_orb_files_by_hash.values())
                print(f"DEBUG: Pre-filter results: {total_skip_subgroups} subgroups skip ORB, {len(need_orb_files_by_hash)} hash groups have {total_need_files} files needing ORB")
                
                # Collect all file pairs that need verification
                # OPTIMIZATION: Two-phase verification with delayed Phase 2 generation
                # Phase 1: Verify first file with all others (generates n-1 pairs)
                # Phase 2: Only verify pairs among files that didn't match reference (dramatically reduces pairs)
                # This reduces verification pairs by 50-90% depending on match rate
                
                # Step 1: Generate Phase 1 pairs only (reference file vs all others)
                phase1_pairs = []
                hash_to_files_map = {}  # Track all files per hash for Phase 2 generation
                
                # 1. Phase 1 pairs within need_orb_files
                for hash_val, files in need_orb_files_by_hash.items():
                    unique_files = list(dict.fromkeys(files))
                    if len(unique_files) > 1:
                        hash_to_files_map[hash_val] = unique_files
                        if len(unique_files) == 2:
                            # Only 2 files, verify the pair
                            phase1_pairs.append((hash_val, unique_files[0], unique_files[1], unique_files, 'phase1'))
                        else:
                            # Phase 1: Verify first file (reference) against all others
                            reference_file = unique_files[0]
                            for candidate_file in unique_files[1:]:
                                phase1_pairs.append((hash_val, reference_file, candidate_file, unique_files, 'phase1'))
                
                # 2. Cross-verify skip_orb_subgroups with need_orb_files
                # This ensures files pre-filtered as duplicates are still verified against files with different sizes
                for hash_val, skip_subgroups in skip_orb_subgroups.items():
                    need_files = need_orb_files_by_hash.get(hash_val, [])
                    
                    if need_files:
                        # For each skip subgroup, verify representative file against all need_files
                        for skip_subgroup in skip_subgroups:
                            # Use first file as representative (all files in subgroup are pre-filtered as duplicates)
                            skip_representative = skip_subgroup[0]
                            
                            for need_file in need_files:
                                # Create a combined group for this cross-verification
                                combined_files = skip_subgroup + need_files
                                phase1_pairs.append((hash_val, skip_representative, need_file, combined_files, 'cross_verify'))
                                # Track files for potential Phase 2
                                if hash_val not in hash_to_files_map:
                                    hash_to_files_map[hash_val] = combined_files
                                else:
                                    # Merge files, avoiding duplicates
                                    existing = set(hash_to_files_map[hash_val])
                                    existing.update(combined_files)
                                    hash_to_files_map[hash_val] = list(existing)
                
                # Remove duplicate pairs from Phase 1
                unique_pairs_set = set()
                unique_phase1_pairs = []
                for pair in phase1_pairs:
                    hash_val, file1, file2, all_files, phase = pair
                    pair_key = tuple(sorted([file1, file2]))
                    if pair_key not in unique_pairs_set:
                        unique_pairs_set.add(pair_key)
                        unique_phase1_pairs.append(pair)
                
                phase1_pairs = unique_phase1_pairs
                print(f"DEBUG: Phase 1 verification pairs: {len(phase1_pairs)}")
                
                if len(phase1_pairs) > 0:
                    # Use union-find to regroup files based on ORB verification
                    verified_groups = {}
                    processed_pairs = 0
                    last_update_time = time.time()
                    
                    # Track which files matched the reference in phase 1
                    reference_matched = {}  # hash_val -> set of files that matched reference
                    unmatched_files = {}  # hash_val -> set of files that didn't match reference
                    
                    # Use parallel processing for ORB verification to speed up processing
                    max_orb_workers = min(4, self.max_workers)  # Limit to 4 workers for ORB to avoid memory issues
                    
                    def verify_pair_wrapper(pair_data):
                        """Wrapper function for parallel verification."""
                        hash_val, file1, file2, all_files, phase = pair_data
                        # Verify pair
                        is_match = self._verify_similarity_opencv(hash_val, hash_val, [file1], [file2], method=self.opencv_verification_method)
                        return (hash_val, file1, file2, all_files, phase, is_match)
                    
                    # Process Phase 1 pairs in parallel batches
                    batch_size = 100  # Process 100 pairs at a time
                    total_phase1_pairs = len(phase1_pairs)
                    
                    for batch_start in range(0, len(phase1_pairs), batch_size):
                        batch = phase1_pairs[batch_start:batch_start + batch_size]
                        
                        # Process batch in parallel
                        with ThreadPoolExecutor(max_workers=max_orb_workers) as executor:
                            batch_results = list(executor.map(verify_pair_wrapper, batch))
                        
                        # Process results
                        for hash_val, file1, file2, all_files, phase, is_match in batch_results:
                            # Initialize group structure for this hash if needed
                            if hash_val not in verified_groups:
                                verified_groups[hash_val] = {f: {f} for f in all_files}
                                reference_matched[hash_val] = set()
                                unmatched_files[hash_val] = set()
                            
                            # Process verification result
                            if is_match:
                                # Merge groups for file1 and file2 (union-find)
                                group1 = verified_groups[hash_val].get(file1, {file1})
                                group2 = verified_groups[hash_val].get(file2, {file2})
                                merged_group = group1 | group2
                                for f in merged_group:
                                    verified_groups[hash_val][f] = merged_group
                                
                                # Track phase 1 matches (when file1 is the reference)
                                if phase == 'phase1' and len(all_files) > 0 and file1 == all_files[0]:
                                    reference_matched[hash_val].add(file2)
                            else:
                                # Track unmatched files for Phase 2
                                if phase == 'phase1' and len(all_files) > 0 and file1 == all_files[0]:
                                    # file2 didn't match reference file1
                                    unmatched_files[hash_val].add(file2)
                        
                        # Update progress after each batch
                        processed_pairs += len(batch)
                        if (time.time() - last_update_time) >= 0.5:
                            progress = processed_pairs / total_phase1_pairs if total_phase1_pairs > 0 else 0.0
                            self.progress_callback(progress * 0.5)  # Phase 1 is 50% of total verification
                            self.status_callback(f"{phase_num}/{total_phases} Scanning: Phase 1 verification {processed_pairs:,}/{total_phase1_pairs:,} pairs...")
                            last_update_time = time.time()
                    
                    # Step 2: Generate Phase 2 pairs only for unmatched files
                    # This dramatically reduces the number of pairs (from O(n²) to O(m²) where m << n)
                    phase2_pairs = []
                    
                    for hash_val, all_files in hash_to_files_map.items():
                        if hash_val not in unmatched_files or len(unmatched_files[hash_val]) < 2:
                            continue
                        
                        # Get unmatched files that need Phase 2 verification
                        unmatched_list = list(unmatched_files[hash_val])
                        
                        # Only generate pairs among unmatched files
                        # This catches subgroups that don't connect through the reference file
                        # Example: If 1-2, 1-3 pass but 1-4, 1-5 fail, we check 4-5
                        for i, file1 in enumerate(unmatched_list):
                            for file2 in unmatched_list[i+1:]:
                                # Check if both files are already in the same group (via union-find)
                                group1 = verified_groups.get(hash_val, {}).get(file1, {file1})
                                group2 = verified_groups.get(hash_val, {}).get(file2, {file2})
                                
                                # Only verify if they're not already grouped
                                if group1 != group2:
                                    phase2_pairs.append((hash_val, file1, file2, all_files, 'phase2'))
                    
                    # Remove duplicate Phase 2 pairs
                    unique_phase2_pairs = []
                    phase2_pairs_set = set()
                    for pair in phase2_pairs:
                        hash_val, file1, file2, all_files, phase = pair
                        pair_key = tuple(sorted([file1, file2]))
                        if pair_key not in phase2_pairs_set:
                            phase2_pairs_set.add(pair_key)
                            unique_phase2_pairs.append(pair)
                    
                    phase2_pairs = unique_phase2_pairs
                    print(f"DEBUG: Phase 2 verification pairs: {len(phase2_pairs)} (only for unmatched files)")
                    
                    # Process Phase 2 pairs if any
                    if len(phase2_pairs) > 0:
                        total_phase2_pairs = len(phase2_pairs)
                        processed_pairs = 0
                        
                        for batch_start in range(0, len(phase2_pairs), batch_size):
                            batch = phase2_pairs[batch_start:batch_start + batch_size]
                            
                            # Process batch in parallel
                            with ThreadPoolExecutor(max_workers=max_orb_workers) as executor:
                                batch_results = list(executor.map(verify_pair_wrapper, batch))
                            
                            # Process results
                            for hash_val, file1, file2, all_files, phase, is_match in batch_results:
                                # Initialize if needed
                                if hash_val not in verified_groups:
                                    verified_groups[hash_val] = {f: {f} for f in all_files}
                                
                                # Process verification result
                                if is_match:
                                    # Merge groups for file1 and file2 (union-find)
                                    group1 = verified_groups[hash_val].get(file1, {file1})
                                    group2 = verified_groups[hash_val].get(file2, {file2})
                                    merged_group = group1 | group2
                                    for f in merged_group:
                                        verified_groups[hash_val][f] = merged_group
                            
                            # Update progress after each batch
                            processed_pairs += len(batch)
                            if (time.time() - last_update_time) >= 0.5:
                                # Phase 2 is the remaining 50% of total verification
                                phase2_progress = processed_pairs / total_phase2_pairs if total_phase2_pairs > 0 else 0.0
                                overall_progress = 0.5 + (phase2_progress * 0.5)
                                self.progress_callback(overall_progress)
                                self.status_callback(f"{phase_num}/{total_phases} Scanning: Phase 2 verification {processed_pairs:,}/{total_phase2_pairs:,} pairs...")
                                last_update_time = time.time()
                    
                    total_pairs = len(phase1_pairs) + len(phase2_pairs)
                    print(f"DEBUG: Total verification pairs: {total_pairs} (Phase 1: {len(phase1_pairs)}, Phase 2: {len(phase2_pairs)})")
                    
                    # Rebuild duplicate_groups from verified groups
                    new_duplicate_groups = {}
                    for hash_val, file_to_group in verified_groups.items():
                        groups = {}
                        seen_files = set()
                        for file, group in file_to_group.items():
                            if file not in seen_files:
                                group_id = min(group)
                                if group_id not in groups:
                                    groups[group_id] = sorted(list(group))
                                seen_files.update(group)
                        
                        # Create groups for all sets with at least 2 files
                        group_count = 0
                        for group_id, group_files in groups.items():
                            if len(group_files) >= 2:
                                subgroup_key = f"{hash_val}_verified_{group_count}"
                                new_duplicate_groups[subgroup_key] = group_files
                                group_count += 1
                    
                    # Merge skip_orb_subgroups (trusted duplicates) with verified groups
                    # Convert skip_orb_subgroups to the format expected by duplicate_groups
                    skip_orb_groups_flat = {}
                    for hash_val, subgroups in skip_orb_subgroups.items():
                        # Each subgroup is a list of files that are pre-filtered as duplicates
                        for subgroup in subgroups:
                            if len(subgroup) > 1:
                                # Use hash_val with a suffix to make it unique
                                group_key = f"{hash_val}_skip_{len(skip_orb_groups_flat)}"
                                skip_orb_groups_flat[group_key] = subgroup
                    
                    duplicate_groups = new_duplicate_groups
                    duplicate_groups.update(skip_orb_groups_flat)  # Add groups that skipped ORB
                    self.progress_callback(1.0)
                    print(f"DEBUG: ORB verification completed: {len(new_duplicate_groups)} verified groups + {len(skip_orb_groups_flat)} trusted groups = {len(duplicate_groups)} total groups")
                else:
                    # No pairs to verify, but we still have skip_orb_subgroups
                    # Convert skip_orb_subgroups to the format expected by duplicate_groups
                    skip_orb_groups_flat = {}
                    for hash_val, subgroups in skip_orb_subgroups.items():
                        for subgroup in subgroups:
                            if len(subgroup) > 1:
                                group_key = f"{hash_val}_skip_{len(skip_orb_groups_flat)}"
                                skip_orb_groups_flat[group_key] = subgroup
                    # duplicate_groups is already initialized earlier in the function
                    duplicate_groups.update(skip_orb_groups_flat)
                    self.progress_callback(1.0)
                    print(f"DEBUG: ORB verification skipped: {len(skip_orb_groups_flat)} trusted groups (no verification needed)")
            
            # Merge pre-filtered exact duplicates (by size/name+timestamp) into final results
            if exact_duplicate_groups:
                duplicate_groups.update(exact_duplicate_groups)
                print(f"DEBUG: Added {len(exact_duplicate_groups)} pre-filtered exact duplicate groups to results")
            
            # Merge RAW/JPEG pairs based on timestamp correlation
            if self.use_imagehash:
                groups_before_merge = len(duplicate_groups)
                duplicate_groups = self._merge_raw_jpeg_by_timestamp(duplicate_groups)
                groups_after_merge = len(duplicate_groups)
                if groups_before_merge != groups_after_merge:
                    print(f"DEBUG: RAW/JPEG merge: {groups_before_merge} -> {groups_after_merge} groups ({groups_before_merge - groups_after_merge} merged)")
            
            # Debug: Count groups by file type
            image_groups = 0
            video_groups = 0
            other_groups = 0
            total_files_in_groups = 0
            for hash_val, files in duplicate_groups.items():
                total_files_in_groups += len(files)
                if files:
                    ext = os.path.splitext(files[0])[1].lower()
                    if ext in self.IMAGE_EXTENSIONS:
                        image_groups += 1
                    elif ext in self.VIDEO_EXTENSIONS:
                        video_groups += 1
                    else:
                        other_groups += 1
            
            # Debug: Cache statistics
            with self._phash_stats_lock:
                phash_lookups = self._phash_cache_lookups
                phash_hits = self._phash_cache_hits
                phash_misses = self._phash_cache_misses
                phash_puts = self._phash_cache_puts
                md5_lookups = self._md5_cache_lookups
                md5_hits = self._md5_cache_hits
                md5_misses = self._md5_cache_misses
                md5_puts = self._md5_cache_puts
            
            print(f"DEBUG: Found {len(duplicate_groups)} duplicate groups")
            print(f"DEBUG:   - Image groups: {image_groups}")
            print(f"DEBUG:   - Video groups: {video_groups}")
            print(f"DEBUG:   - Other groups: {other_groups}")
            print(f"DEBUG:   - Total files in groups: {total_files_in_groups}")
            print(f"DEBUG:   - use_imagehash={self.use_imagehash}, use_multi_hash={self.use_multi_hash}")
            print(f"DEBUG: Cache stats - pHash: {phash_lookups} lookups, {phash_hits} hits, {phash_misses} misses, {phash_puts} puts")
            if phash_lookups > 0:
                hit_rate = (phash_hits / phash_lookups) * 100
                print(f"DEBUG: pHash cache hit rate: {hit_rate:.1f}%")
            print(f"DEBUG: Cache stats - MD5: {md5_lookups} lookups, {md5_hits} hits, {md5_misses} misses, {md5_puts} puts")
            print(f"DEBUG: Cache DB path: {self._phash_db_path}")
            print(f"DEBUG: Hash mode - use_imagehash: {self.use_imagehash}, use_multi_hash: {self.use_multi_hash}")
            
            self.file_groups_raw = duplicate_groups
            print(f"DEBUG: Calling results_callback with {len(duplicate_groups)} groups")
            self.results_callback(duplicate_groups)
            print(f"DEBUG: results_callback completed")
            
        except Exception as e:
            print(f"Scan error: {e}")
            import traceback
            traceback.print_exc()
            self.results_callback({})
    
    def apply_sorting(
        self,
        file_groups: Dict[str, List[str]],
        sort_value: str,
        group_by_type: bool = False
    ) -> Dict[str, List[str]]:
        """Apply sorting to file groups."""
        def get_sort_key(item: Tuple[str, List[str]]):
            hash_val, files = item
            file_count = len(files)
            
            total_size = 0
            file_names = []
            mtimes = []
            
            for fp in files:
                try:
                    stat = os.stat(fp)
                    total_size += stat.st_size
                    mtimes.append(stat.st_mtime)
                    file_names.append(os.path.basename(fp).lower())
                except Exception:
                    pass
            
            type_key = ""
            if group_by_type and files:
                ext = os.path.splitext(files[0])[1].lower()
                if ext in self.IMAGE_EXTENSIONS:
                    type_key = "1_Images"
                elif ext in self.VIDEO_EXTENSIONS:
                    type_key = "2_Videos"
                elif ext in self.AUDIO_EXTENSIONS:
                    type_key = "3_Audio"
                elif ext in self.PDF_EXTENSIONS or ext in self.EPUB_EXTENSIONS:
                    type_key = "4_Documents"
                else:
                    type_key = "5_Others"
            
            if sort_value == "Count (High-Low)":
                sec_key = (-file_count, -total_size)
            elif sort_value == "Count (Low-High)":
                sec_key = (file_count, total_size)
            elif sort_value in ["Size (High-Low)", "File Size (Large-Small)"]:
                sec_key = (-total_size, -file_count)
            elif sort_value in ["Size (Low-High)", "File Size (Small-Large)"]:
                sec_key = (total_size, file_count)
            elif sort_value == "Name (A-Z)":
                sec_key = (sorted(file_names)[0] if file_names else "",)
            elif sort_value == "Name (Z-A)":
                sec_key = (sorted(file_names, reverse=True)[0] if file_names else "",)
            elif sort_value == "Newest First":
                sec_key = (-max(mtimes) if mtimes else 0,)
            elif sort_value == "Oldest First":
                sec_key = (min(mtimes) if mtimes else 0,)
            else:
                sec_key = (-file_count, -total_size)
            
            return (type_key,) + sec_key
        
        sorted_items = sorted(file_groups.items(), key=get_sort_key)
        return dict(sorted_items)
    
    def compute_quick_select_decisions(
        self,
        file_groups: Dict[str, List[str]],
        mode: str,
        apply_all: bool = False,
        current_page_groups: Optional[List[Tuple[str, List[str]]]] = None,
    ) -> Dict[str, bool]:
        """
        Compute quick select decisions (which files to keep/delete).
        
        Returns:
            Dict mapping file_path -> should_delete (True means mark for deletion)
        """
        decisions: Dict[str, bool] = {}
        
        # Determine scope
        if apply_all or current_page_groups is None:
            items = list(file_groups.items())
        else:
            items = current_page_groups
        
        if mode == 'newest':
            for hash_val, files in items:
                if len(files) < 2:
                    continue
                file_details = []
                for fp in files:
                    try:
                        stat = os.stat(fp)
                        file_details.append({
                            'path': fp,
                            'mtime': stat.st_mtime,
                        })
                    except Exception:
                        pass
                if not file_details:
                    continue
                keep_path = max(file_details, key=lambda x: x['mtime'])['path']
                keep_mtime = next(f['mtime'] for f in file_details if f['path'] == keep_path)
                candidates = [f['path'] for f in file_details if f['mtime'] == keep_mtime]
                if len(candidates) > 1:
                    keep_path = sorted(candidates, key=lambda p: (len(p), p))[0]
                if keep_path not in files:
                    keep_path = files[0]
                for fp in files:
                    decisions[fp] = (fp != keep_path)
        
        elif mode == 'oldest':
            for hash_val, files in items:
                if len(files) < 2:
                    continue
                file_details = []
                for fp in files:
                    try:
                        stat = os.stat(fp)
                        file_details.append({
                            'path': fp,
                            'mtime': stat.st_mtime,
                        })
                    except Exception:
                        pass
                if not file_details:
                    continue
                keep_path = min(file_details, key=lambda x: x['mtime'])['path']
                keep_mtime = next(f['mtime'] for f in file_details if f['path'] == keep_path)
                candidates = [f['path'] for f in file_details if f['mtime'] == keep_mtime]
                if len(candidates) > 1:
                    keep_path = sorted(candidates, key=lambda p: (len(p), p))[0]
                if keep_path not in files:
                    keep_path = files[0]
                for fp in files:
                    decisions[fp] = (fp != keep_path)
        
        elif mode == 'best_res':
            # Keep highest resolution image (W*H), if multiple images share the highest resolution, keep the one with largest file size
            for hash_val, files in items:
                if len(files) < 2:
                    continue
                img_files = [fp for fp in files if self.is_image_file(fp)]
                if len(img_files) < 2:
                    continue
                details = []
                for fp in img_files:
                    try:
                        st = os.stat(fp)
                        size_bytes = int(getattr(st, "st_size", 0) or 0)
                        mtime = float(getattr(st, "st_mtime", 0.0) or 0.0)
                        w = h = 0
                        try:
                            with Image.open(fp) as im:
                                w, h = im.size
                        except Exception:
                            w = h = 0
                        # Calculate resolution as W * H (area)
                        resolution = int(w * h)
                        details.append({
                            "path": fp,
                            "resolution": resolution,
                            "width": int(w),
                            "height": int(h),
                            "size": size_bytes,
                            "mtime": mtime,
                        })
                    except Exception:
                        continue
                if not details:
                    continue
                
                # Step 1: Find the highest resolution (W*H)
                max_resolution = max(d["resolution"] for d in details)
                
                # Step 2: Filter to only images with the highest resolution
                highest_res_images = [d for d in details if d["resolution"] == max_resolution]
                
                # Step 3: Among images with highest resolution, keep the one with largest file size
                # (better quality, less compression)
                # If file sizes are equal, use mtime, path length, and path as tiebreakers
                keep = max(
                    highest_res_images,
                    key=lambda d: (d["size"], -d["mtime"], -len(d["path"]), d["path"])
                )["path"]
                
                if keep not in img_files and img_files:
                    keep = img_files[0]
                for fp in img_files:
                    decisions[fp] = (fp != keep)
        
        elif mode == 'keep_smallest':
            # Keep highest resolution image (W*H), if multiple images share the highest resolution, keep the one with smallest file size
            for hash_val, files in items:
                if len(files) < 2:
                    continue
                img_files = [fp for fp in files if self.is_image_file(fp)]
                if len(img_files) < 2:
                    continue
                details = []
                for fp in img_files:
                    try:
                        st = os.stat(fp)
                        size_bytes = int(getattr(st, "st_size", 0) or 0)
                        mtime = float(getattr(st, "st_mtime", 0.0) or 0.0)
                        w = h = 0
                        try:
                            with Image.open(fp) as im:
                                w, h = im.size
                        except Exception:
                            w = h = 0
                        # Calculate resolution as W * H (area)
                        resolution = int(w * h)
                        details.append({
                            "path": fp,
                            "resolution": resolution,
                            "width": int(w),
                            "height": int(h),
                            "size": size_bytes,
                            "mtime": mtime,
                        })
                    except Exception:
                        continue
                if not details:
                    continue
                
                # Step 1: Find the highest resolution (W*H)
                max_resolution = max(d["resolution"] for d in details)
                
                # Step 2: Filter to only images with the highest resolution
                highest_res_images = [d for d in details if d["resolution"] == max_resolution]
                
                # Step 3: Among images with highest resolution, keep the one with smallest file size
                # If file sizes are equal, use mtime, path length, and path as tiebreakers
                keep = min(
                    highest_res_images,
                    key=lambda d: (d["size"], d["mtime"], len(d["path"]), d["path"])
                )["path"]
                
                if keep not in img_files and img_files:
                    keep = img_files[0]
                for fp in img_files:
                    decisions[fp] = (fp != keep)
        
        elif mode == 'keep_raw':
            # Keep RAW file if present, only applicable when RAW and non-RAW files are mixed
            for hash_val, files in items:
                if len(files) < 2:
                    continue
                img_files = [fp for fp in files if self.is_image_file(fp)]
                if len(img_files) < 2:
                    continue
                raw_files = [fp for fp in img_files if self.is_raw_image_file(fp)]
                non_raw_files = [fp for fp in img_files if not self.is_raw_image_file(fp)]
                
                # Only process groups that have both RAW and non-RAW files
                if raw_files and non_raw_files:
                    # Keep best RAW
                    details = []
                    for fp in raw_files:
                        try:
                            st = os.stat(fp)
                            mtime = float(getattr(st, "st_mtime", 0.0) or 0.0)
                            w = h = 0
                            try:
                                with Image.open(fp) as im:
                                    w, h = im.size
                            except Exception:
                                pass
                            maxd = int(max(w, h)) if w and h else 0
                            mind = int(min(w, h)) if w and h else 0
                            area = int(maxd * mind)
                            details.append({
                                "path": fp,
                                "area": area,
                                "maxd": maxd,
                                "mtime": mtime,
                            })
                        except Exception:
                            continue
                    if details:
                        keep = max(details, key=lambda d: (d["area"], -d["mtime"]))["path"]
                    else:
                        keep = raw_files[0]
                    # Mark all non-RAW files for deletion, keep the best RAW
                    for fp in img_files:
                        if fp in raw_files:
                            decisions[fp] = (fp != keep)  # Delete other RAW files, keep the best one
                        else:
                            decisions[fp] = True  # Delete all non-RAW files
                # If no RAW files or no mixed RAW/non-RAW, skip this group (don't make any decisions)
        
        return decisions



