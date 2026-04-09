"""
Persistent Thumbnail Cache for CloneWiper.
Stores generated thumbnails in a SQLite database to improve performance on subsequent runs.
"""
import logging
import sqlite3
import threading
import os
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class PersistentThumbnailCache:
    """
    SQLite-backed persistent cache for thumbnails.
    """
    def __init__(self, db_path: str = "thumbnails.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._ensure_db()

    def _get_platform_db_path(self) -> str:
        """Get appropriate cache path for the platform."""
        try:
            import platform
            system = platform.system()
            if system == "Windows":
                base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
                if base:
                    cache_dir = os.path.join(base, "CloneWiper")
                    os.makedirs(cache_dir, exist_ok=True)
                    return os.path.join(cache_dir, "thumbnails.db")
            # Fallback for other OS or if env vars missing
            home = os.path.expanduser("~")
            cache_dir = os.path.join(home, ".CloneWiper")
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, "thumbnails.db")
        except Exception:
            return "thumbnails.db"

    def _ensure_db(self):
        """Initialize the database and tables."""
        if self.db_path == "thumbnails.db":
            self.db_path = self._get_platform_db_path()
            
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS thumbnails (
                        path TEXT PRIMARY KEY,
                        mtime_ns INTEGER NOT NULL,
                        size INTEGER NOT NULL,
                        width INTEGER NOT NULL,
                        height INTEGER NOT NULL,
                        data BLOB NOT NULL,
                        last_access INTEGER NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_last_access ON thumbnails(last_access)")
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug("Error initializing thumbnail cache: %s", e)

    def get(self, path: str, mtime_ns: int, size: int) -> Optional[bytes]:
        """
        Retrieve thumbnail from cache if it matches path, mtime, and size.
        Returns raw JPEG bytes or None.
        """
        try:
            # Normalize path
            abs_path = os.path.abspath(path)
            
            # Use a fresh connection per thread/request to be safe or managed carefully
            # For simplicity in this helper, we'll open/close. 
            # In high-perf scenarios, connection pooling is better, but SQLite fits 
            # effectively with short-lived connections in WAL mode.
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data FROM thumbnails WHERE path=? AND mtime_ns=? AND size=?",
                    (abs_path, mtime_ns, size)
                )
                row = cursor.fetchone()
                if row:
                    # Update last access time asynchronously or lazily?
                    # For now, let's update it to implement LRU later if needed
                    # We do it successfully but don't block return on it if it fails
                    try: 
                        conn.execute(
                            "UPDATE thumbnails SET last_access=? WHERE path=?",
                            (int(time.time()), abs_path)
                        )
                    except: 
                        pass
                    return row[0]
        except Exception as e:
            # print(f"Cache miss/error: {e}")
            pass
        return None

    def put(self, path: str, mtime_ns: int, size: int, width: int, height: int, data: bytes):
        """Store thumbnail in cache."""
        try:
            abs_path = os.path.abspath(path)
            now = int(time.time())
            
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO thumbnails (path, mtime_ns, size, width, height, data, last_access)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (abs_path, mtime_ns, size, width, height, data, now)
                )
        except Exception as e:
            logger.debug("Error saving to thumbnail cache: %s", e)

    def cleanup(self, max_days=30):
        """Remove old entries."""
        try:
            cutoff = int(time.time()) - (max_days * 86400)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM thumbnails WHERE last_access < ?", (cutoff,))
                conn.execute("VACUUM")
        except Exception as e:
            logger.debug("Error cleaning thumbnail cache: %s", e)
