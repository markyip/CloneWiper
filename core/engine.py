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
from typing import Dict, List, Set, Optional, Callable, Tuple
import sys

# Try to import psutil for advanced CPU detection (hybrid architecture support)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional dependencies
try:
    import imagehash
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
        
        # Results
        self.file_groups: Dict[str, List[str]] = defaultdict(list)
        self.file_groups_raw: Dict[str, List[str]] = {}
        
        # Cache settings
        self.phash_cache_enabled = True
        self.md5_cache_enabled = True
        self.use_imagehash = False
        self.use_multi_hash = False  # True for multi-algorithm, False for single algorithm
        
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
        
        Args:
            img: PIL Image object
            
        Returns:
            Hash string or None on error
        """
        try:
            if not IMAGEHASH_AVAILABLE:
                return None
            
            # Convert to RGB if needed
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception as convert_error:
                print(f"DEBUG: Image mode conversion failed, using original mode: {convert_error}")
            
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
    
    def _calculate_multi_hash(self, img: Image.Image) -> Optional[str]:
        """
        Calculate combined perceptual hash using multiple algorithms for better accuracy.
        Uses: average_hash, phash (perceptual), dhash (difference), and whash (wavelet).
        
        Optimizations:
        - Pre-resize image to smaller size (256x256 max) for faster processing
        - Convert to RGB once and reuse
        - Parallel calculation of hash algorithms using ThreadPoolExecutor
        
        Args:
            img: PIL Image object
            
        Returns:
            Combined hash string in format: "avg_phash_diff_wave" or None on error
        """
        try:
            if not IMAGEHASH_AVAILABLE:
                return None
            
            # Optimization: Convert to RGB once (if needed) and reuse
            # Note: Image resizing is already done during image loading for better performance
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception as convert_error:
                # If conversion fails, try to continue with original mode
                print(f"DEBUG: Image mode conversion failed, using original mode: {convert_error}")
            
            # Optimization 3: Parallel calculation of hash algorithms
            # Use ThreadPoolExecutor to calculate all hashes concurrently
            hash_results = {}
            
            def calc_hash(algorithm_name, hash_func):
                """Helper function to calculate a single hash with error handling."""
                try:
                    return algorithm_name, str(hash_func(img))
                except Exception as e:
                    print(f"DEBUG: {algorithm_name} failed: {e}")
                    return algorithm_name, None
            
            # Calculate all hashes in parallel (up to 4 concurrent operations)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(calc_hash, 'average', imagehash.average_hash),
                    executor.submit(calc_hash, 'perceptual', imagehash.phash),
                    executor.submit(calc_hash, 'difference', imagehash.dhash),
                    executor.submit(calc_hash, 'wavelet', imagehash.whash),
                ]
                
                for future in as_completed(futures):
                    try:
                        algo_name, hash_value = future.result()
                        hash_results[algo_name] = hash_value
                    except Exception as e:
                        print(f"DEBUG: Hash calculation future error: {e}")
            
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
                            return f"img_{cached}"
                        else:
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
                                print(f"DEBUG: RAW file processing failed for {file_path}: {raw_error}")
                                pass
                        else:
                            # Handle common image formats
                            # ImageFile.LOAD_TRUNCATED_IMAGES is already set to True at module level
                            # This allows loading truncated images (common with some JPEG files)
                            try:
                                img = Image.open(file_path)
                                # Optimization: For large images, use thumbnail loading for faster processing
                                # This is especially beneficial for multi-hash calculation
                                if self.use_multi_hash:
                                    # Get image size without fully loading (fast operation)
                                    w, h = img.size
                                    max_dimension = 256
                                    if w > max_dimension or h > max_dimension:
                                        # For large images: thumbnail is MUCH faster than processing full size
                                        # LANCZOS provides good quality but is slower. For hash calculation,
                                        # we can use faster resampling methods (NEAREST or BILINEAR) for better performance
                                        # However, LANCZOS is still acceptable for the quality/speed tradeoff
                                        # The thumbnail operation is in-place and memory-efficient
                                        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                                        # Note: thumbnail() is optimized and only loads what's needed for the final size
                                    else:
                                        # Small image (≤256x256): no resizing needed, just load normally
                                        img.load()
                                else:
                                    # Single hash mode - load normally (no resizing needed for single hash)
                                    img.load()
                            except Exception as img_open_error:
                                # If loading fails, try opening without load() call
                                # Some truncated images can still be processed
                                try:
                                    img = Image.open(file_path)
                                    # Apply thumbnail optimization if multi-hash
                                    if self.use_multi_hash:
                                        w, h = img.size
                                        max_dimension = 256
                                        if w > max_dimension or h > max_dimension:
                                            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                                except Exception:
                                    # Image cannot be opened, will fall back to MD5
                                    print(f"DEBUG: Cannot open image file {file_path}: {img_open_error}")
                                    img = None
                        
                        if img is not None:
                            # Use single or multi-hash algorithm based on setting
                            if self.use_multi_hash:
                                img_hash = self._calculate_multi_hash(img)
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
                    except Exception as img_error:
                        # Image processing failed, fall back to MD5
                        print(f"DEBUG: Image hash calculation failed for {file_path}: {img_error}")
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
                    
                    if total and processed - last_ui >= 2000:
                        last_ui = processed
                        # Calculate phase progress (0.0-1.0) for current phase only
                        phase_progress = processed / total if total > 0 else 0.0
                        # Pass phase progress (0.0-1.0) instead of overall progress
                        self.progress_callback(phase_progress)
                        self.status_callback(f"{phase_num}/{total_phases} Scanning: Pre-filtering (Partial Hash) {processed:,}/{total:,}")
                    
                    while len(inflight) < max_inflight and submit_one():
                        pass
        
        return partial_groups
    
    def _group_by_similarity_multi_hash(self, hash_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Group files by multi-algorithm hash similarity using hamming distance.
        
        Uses a voting mechanism: requires at least 3 out of 4 algorithms to agree
        that images are similar (hamming distance <= threshold).
        
        This reduces false positives compared to exact match while maintaining
        good accuracy for true duplicates.
        
        Args:
            hash_groups: Dict mapping hash strings to file lists
            
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
        
        # Hamming distance threshold for each algorithm (64-bit hashes)
        # Typical thresholds: 5-10 for similar images, stricter for duplicates
        # We use 8 as a balance between accuracy and false positive reduction
        HAMMING_THRESHOLD = 8
        
        # Voting mechanism: require at least 3 out of 4 algorithms to agree
        MIN_AGREEMENT = 3
        
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
        exact_match_groups = {}
        for hash_val, files in hash_list:
            if hash_val not in processed and len(files) > 1:
                unique_files = list(dict.fromkeys(files))
                if len(unique_files) > 1:
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
            
            # Phase 1: Find candidates using average_hash only (fast)
            candidate_pairs = set()  # (hash1, hash2) pairs that might be similar
            avg_hash_list = list(avg_hash_index.items())
            for i, (key1, hashes1) in enumerate(avg_hash_list):
                for hash1, files1, h_avg1 in hashes1:
                    # Compare with other average hashes in nearby buckets
                    for j, (key2, hashes2) in enumerate(avg_hash_list[i:], start=i):
                        for hash2, files2, h_avg2 in hashes2:
                            if hash1 == hash2:
                                continue
                            try:
                                if h_avg1 - h_avg2 <= HAMMING_THRESHOLD:
                                    # Potential match, add to candidates
                                    if hash1 < hash2:
                                        candidate_pairs.add((hash1, hash2))
                                    else:
                                        candidate_pairs.add((hash2, hash1))
                            except Exception:
                                pass
            
            # Phase 2: Detailed comparison only for candidate pairs
            # Use parallel processing for detailed comparison
            def compare_pair(pair):
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
                    
                    # Check all algorithms
                    if h_avg1 and h_avg2 and (h_avg1 - h_avg2) <= HAMMING_THRESHOLD:
                        agreements += 1
                    if h_phash1 and h_phash2 and (h_phash1 - h_phash2) <= HAMMING_THRESHOLD:
                        agreements += 1
                    if h_dhash1 and h_dhash2 and (h_dhash1 - h_dhash2) <= HAMMING_THRESHOLD:
                        agreements += 1
                    if h_whash1 and h_whash2 and (h_whash1 - h_whash2) <= HAMMING_THRESHOLD:
                        agreements += 1
                    
                    if agreements >= MIN_AGREEMENT:
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
                
                # Use ThreadPoolExecutor for parallel comparison
                with ThreadPoolExecutor(max_workers=min(8, self.max_workers)) as executor:
                    results = executor.map(compare_pair, candidate_list)
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
    
    def scan_duplicate_files(self, scan_paths: List[str], use_imagehash: bool = False, use_multi_hash: bool = True):
        """
        Main scanning function.
        
        Args:
            scan_paths: List of directory paths to scan
            use_imagehash: Whether to use perceptual hashing for images and videos
            use_multi_hash: If True, use multi-algorithm hash; if False, use single algorithm (phash)
        """
        try:
            self.scan_cancelled = False
            self.use_imagehash = use_imagehash and IMAGEHASH_AVAILABLE
            self.use_multi_hash = use_multi_hash if self.use_imagehash else False
            
            # Debug: Log hash mode and availability
            print(f"DEBUG: scan_duplicate_files called with use_imagehash={use_imagehash}, IMAGEHASH_AVAILABLE={IMAGEHASH_AVAILABLE}")
            print(f"DEBUG: Final settings: use_imagehash={self.use_imagehash}, use_multi_hash={self.use_multi_hash}")
            
            # Check OpenCV availability for video hashing
            try:
                import cv2
                OPENCV_AVAILABLE = True
            except ImportError:
                OPENCV_AVAILABLE = False
            print(f"DEBUG: OpenCV available for video hashing: {OPENCV_AVAILABLE}")
            
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
            total_phases = 4 if not self.use_imagehash else 3  # Phase 3 skipped if using imagehash
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
            
            if self.use_imagehash:
                image_files_for_hash = list(dict.fromkeys(all_image_files))
                non_image_files_by_size = []
                
                for size, files in size_groups.items():
                    if len(files) > 1:
                        non_image_in_group = [fp for fp in files if not self.is_image_file(fp)]
                        if len(non_image_in_group) > 1:
                            non_image_files_by_size.extend(non_image_in_group)
                
                if len(image_files_for_hash) > 1:
                    potential_duplicates_by_size.extend(image_files_for_hash)
                
                potential_duplicates_by_size.extend(non_image_files_by_size)
            else:
                for size, files in size_groups.items():
                    if len(files) > 1:
                        potential_duplicates_by_size.extend(files)
            
            print(f"DEBUG: Found {len(potential_duplicates_by_size)} potential duplicate files")
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
                # Use hash_workers (more threads) for I/O-bound hash calculation
                with ThreadPoolExecutor(max_workers=self.hash_workers) as executor:
                    future_to_file = {
                        executor.submit(self.calculate_file_hash, file_path): file_path
                        for file_path in files_to_full_hash
                    }
                    
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
                        self.files_scanned = processed
                        
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
            
            # Filter duplicate groups
            # For multi-algorithm hash, use hamming distance comparison instead of exact match
            duplicate_groups = {}
            
            if self.use_multi_hash:
                # Multi-algorithm hash: Use hamming distance with voting mechanism
                # This reduces false positives by requiring multiple algorithms to agree
                duplicate_groups = self._group_by_similarity_multi_hash(hash_groups)
            else:
                # Single hash or MD5: Use exact match (original behavior)
                for hash_val, files in hash_groups.items():
                    if len(files) > 1:
                        unique_files = list(dict.fromkeys(files))
                        if len(unique_files) > 1:
                            duplicate_groups[hash_val] = unique_files
            
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



