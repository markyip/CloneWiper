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
    file_type: str  # 'image', 'video', 'audio', 'pdf', 'epub', 'other'
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
        self.max_workers = min(8, os.cpu_count() or 4)
        self.hash_lock = threading.Lock()
        
        # Results
        self.file_groups: Dict[str, List[str]] = defaultdict(list)
        self.file_groups_raw: Dict[str, List[str]] = {}
        
        # Cache settings
        self.phash_cache_enabled = True
        self.md5_cache_enabled = True
        self.use_imagehash = False
        
        # Persistent cache DB
        self._phash_db = None
        self._phash_db_lock = threading.Lock()
        self._phash_db_path = self._get_default_phash_db_path()
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
        """Choose a writable cache location (cross-platform)."""
        import platform
        
        # Try platform-specific cache directories
        try:
            system = platform.system()
            if system == "Windows":
                base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
                if base:
                    cache_dir = os.path.join(base, "CloneWiper")
                    os.makedirs(cache_dir, exist_ok=True)
                    return os.path.join(cache_dir, "phash_cache.sqlite3")
            elif system == "Darwin":  # macOS
                cache_dir = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "CloneWiper")
                os.makedirs(cache_dir, exist_ok=True)
                return os.path.join(cache_dir, "phash_cache.sqlite3")
        except Exception:
            pass
        
        # Fallback to project directory
        try:
            cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, "phash_cache.sqlite3")
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
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS phash_cache (
                        path TEXT PRIMARY KEY,
                        size INTEGER NOT NULL,
                        mtime_ns INTEGER NOT NULL,
                        algo TEXT NOT NULL,
                        hash TEXT NOT NULL,
                        updated INTEGER NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_phash_updated ON phash_cache(updated)")
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
            row = conn.execute(
                "SELECT size, mtime_ns, algo, hash FROM phash_cache WHERE path=?",
                (p,),
            ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        try:
            s0, m0, a0, h0 = row
            if int(s0) == int(size) and int(m0) == int(mtime_ns) and str(a0) == str(algo):
                return str(h0)
        except Exception:
            return None
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
                    ON CONFLICT(path) DO UPDATE SET
                      size=excluded.size,
                      mtime_ns=excluded.mtime_ns,
                      algo=excluded.algo,
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
    
    def _calculate_multi_hash(self, img: Image.Image) -> Optional[str]:
        """
        Calculate combined perceptual hash using multiple algorithms for better accuracy.
        Uses: average_hash, phash (perceptual), dhash (difference), and whash (wavelet).
        
        Args:
            img: PIL Image object
            
        Returns:
            Combined hash string in format: "avg_phash_diff_wave" or None on error
        """
        try:
            if not IMAGEHASH_AVAILABLE:
                return None
            
            # Convert to RGB if needed (required for some hash algorithms)
            # Use a copy to avoid modifying the original image
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception as convert_error:
                # If conversion fails, try to continue with original mode
                print(f"DEBUG: Image mode conversion failed, using original mode: {convert_error}")
            
            # Calculate multiple hash types with individual error handling
            # This allows partial success if some algorithms fail
            hash_average = None
            hash_perceptual = None
            hash_difference = None
            hash_wavelet = None
            
            # Try each hash algorithm individually
            try:
                hash_average = str(imagehash.average_hash(img))
            except Exception as e:
                print(f"DEBUG: average_hash failed: {e}")
            
            try:
                hash_perceptual = str(imagehash.phash(img))  # More accurate than average_hash
            except Exception as e:
                print(f"DEBUG: phash failed: {e}")
            
            try:
                hash_difference = str(imagehash.dhash(img))  # Sensitive to brightness changes
            except Exception as e:
                print(f"DEBUG: dhash failed: {e}")
            
            try:
                hash_wavelet = str(imagehash.whash(img))     # Wavelet-based hash
            except Exception as e:
                print(f"DEBUG: whash failed: {e}")
            
            # Combine available hashes (use placeholder for failed ones)
            hashes = []
            if hash_average:
                hashes.append(hash_average)
            else:
                hashes.append("")
            
            if hash_perceptual:
                hashes.append(hash_perceptual)
            else:
                hashes.append("")
            
            if hash_difference:
                hashes.append(hash_difference)
            else:
                hashes.append("")
            
            if hash_wavelet:
                hashes.append(hash_wavelet)
            else:
                hashes.append("")
            
            # If at least one hash succeeded, return combined hash
            if any(hashes):
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
                    
                    algo = "multi_hash"  # Updated algorithm name
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
                                # Try to load the image to verify it's readable
                                # This will work even for truncated images if LOAD_TRUNCATED_IMAGES is True
                                img.load()
                            except Exception as img_open_error:
                                # If loading fails, try opening without load() call
                                # Some truncated images can still be processed
                                try:
                                    img = Image.open(file_path)
                                except Exception:
                                    # Image cannot be opened, will fall back to MD5
                                    print(f"DEBUG: Cannot open image file {file_path}: {img_open_error}")
                                    img = None
                        
                        if img is not None:
                            # Use multi-hash algorithm for better accuracy
                            img_hash = self._calculate_multi_hash(img)
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
            
            algo = "video_multi_hash"  # Updated algorithm name
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
                print(f"DEBUG: OpenCV not available for video hash, falling back to MD5 for {file_path}")
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
                            frame_hash = self._calculate_multi_hash(img)
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
        """Calculate partial MD5 hash (Start + Middle + End)."""
        try:
            file_size = os.path.getsize(file_path)
            with open(file_path, 'rb') as f:
                data = f.read(chunk_size)
                if not data:
                    return None
                if file_size > chunk_size * 3:
                    f.seek(file_size // 2)
                    data += f.read(chunk_size)
                    f.seek(-chunk_size, 2)
                    data += f.read(chunk_size)
                return hashlib.md5(data).hexdigest()
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
                                    if entry.name.lower().endswith('.txt'):
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
    
    def perform_partial_hashing(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Perform partial hashing for fast pre-filtering."""
        partial_groups = defaultdict(list)
        failed_count = 0
        total = len(file_paths)
        processed = 0
        
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
        
        max_workers = max(2, int(self.max_workers * 2))
        max_inflight = max(32, max_workers * 8)
        
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
                        self.status_callback(f"Pre-filtering (Partial Hash): {processed:,}/{total:,}")
                    
                    while len(inflight) < max_inflight and submit_one():
                        pass
        
        return partial_groups
    
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
    
    def scan_duplicate_files(self, scan_paths: List[str], use_imagehash: bool = False):
        """
        Main scanning function.
        
        Args:
            scan_paths: List of directory paths to scan
            use_imagehash: Whether to use perceptual hashing for images and videos
        """
        try:
            self.scan_cancelled = False
            self.use_imagehash = use_imagehash and IMAGEHASH_AVAILABLE
            
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
            
            if self.use_imagehash:
                print("DEBUG: Calling status_callback: Collecting files (ImageHash enabled)...")
                self.status_callback("Collecting files (ImageHash enabled)...")
            else:
                print("DEBUG: Calling status_callback: Collecting files...")
                self.status_callback("Collecting files...")
            
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
                                        if entry.name.startswith('._') or entry.name.lower().endswith('.txt'):
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
                for future in as_completed(futures):
                    if self.scan_cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        return
                    try:
                        future.result()
                        # Update status periodically during collection
                        with total_collected_lock:
                            count = total_collected[0]
                        if count - last_status_update >= 1000:
                            self.status_callback(f"Collected {count:,} files...")
                            last_status_update = count
                    except Exception as e:
                        print(f"Scan task error: {e}")
                        continue
                
                # Final status update
                with total_collected_lock:
                    final_count = total_collected[0]
                with total_image_files_lock:
                    img_count = total_image_files[0]
                if img_count > 0:
                    self.status_callback(f"Collected {final_count:,} files ({img_count:,} images)")
                else:
                    self.status_callback(f"Collected {final_count:,} files")
            
            # Phase 2: Filter potential duplicates
            print(f"DEBUG: Phase 2 - Filtering potential duplicates")
            self.status_callback("Filtering potential duplicates...")
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
                    partial_groups = self.perform_partial_hashing(non_image_files)
                    for files in partial_groups.values():
                        if len(files) > 1:
                            files_to_full_hash.extend(files)
            else:
                self.status_callback("Pre-filtering (Partial Hash)...")
                partial_groups = self.perform_partial_hashing(potential_duplicates_by_size)
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
            
            # Phase 4: Full hash calculation
            hash_groups = defaultdict(list)
            processed = 0
            progress_update_interval = max(1, self.total_files // 100)
            
            # CPU processing
            if files_to_full_hash:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
                        
                        if processed % progress_update_interval == 0 or processed == self.total_files:
                            progress_value = processed / self.total_files if self.total_files > 0 else 0.0
                            self.progress_callback(progress_value)
                            self.status_callback(f"Hashing: {processed}/{self.total_files}")
            
            # Filter duplicate groups
            duplicate_groups = {}
            for hash_val, files in hash_groups.items():
                if len(files) > 1:
                    unique_files = list(dict.fromkeys(files))
                    if len(unique_files) > 1:
                        duplicate_groups[hash_val] = unique_files
            
            # Merge RAW/JPEG pairs based on timestamp correlation
            if self.use_imagehash:
                duplicate_groups = self._merge_raw_jpeg_by_timestamp(duplicate_groups)
            
            print(f"DEBUG: Found {len(duplicate_groups)} duplicate groups")
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
            # Keep highest resolution image
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
                        maxd = int(max(w, h))
                        mind = int(min(w, h))
                        area = int(maxd * mind)
                        details.append({
                            "path": fp,
                            "area": area,
                            "maxd": maxd,
                            "mind": mind,
                            "size": size_bytes,
                            "mtime": mtime,
                        })
                    except Exception:
                        continue
                if not details:
                    continue
                # Sort by: area (desc), maxd (desc), mind (desc), size (asc), mtime (asc), path length (asc), path (asc)
                # When all other factors are equal, prefer shorter path (consistent with Keep Newest/Oldest)
                keep = min(
                    details,
                    key=lambda d: (-d["area"], -d["maxd"], -d["mind"], d["size"], d["mtime"], len(d["path"]), d["path"])
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



