"""
CloneWiper - Qt UI Implementation

Modern UI implementation using PySide6 with Material Design 3 styling.
Qt UI provides excellent performance when handling large result sets and supports rich interactive features.

Key Features:
    - Material Design 3 styled interface
    - Horizontal comparison strip for each duplicate group
    - Asynchronous thumbnail loading (using QThreadPool)
    - Thread-safe signal/slot mechanism
    - Pagination for large result sets
    - Browse path memory functionality

Dependencies:
    - PySide6 >= 6.5.0
    - Pillow (image processing)
    - ImageHash (perceptual hashing, optional)
    - OpenCV (video thumbnails, optional)
    - PyMuPDF (PDF thumbnails, optional)

Author: Mark Yip
Version: 2.0
"""
import sys
import os
import logging
import ctypes
import ctypes.wintypes
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import time
import threading

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import (  # type: ignore[reportMissingImports]
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QCheckBox, QComboBox, QScrollArea,
        QFrame, QListView, QStyledItemDelegate, QStyleOptionViewItem,
        QSizePolicy, QMessageBox, QFileDialog, QProgressBar, QGridLayout, QScrollBar,
        QListWidget, QListWidgetItem, QDialogButtonBox, QDialog, QStyleOptionComboBox, QStyle
    )
    from PySide6.QtCore import (  # type: ignore[reportMissingImports]
        Qt, QSize, QThread, QThreadPool, QRunnable, Signal, QObject, QModelIndex,
        QAbstractListModel, QRect, QPoint, QTimer, QMutex, QWaitCondition, Slot, QSettings,
        QVariantAnimation, QEasingCurve, QPropertyAnimation, QEvent, QBuffer
    )
    from PySide6.QtGui import (  # type: ignore[reportMissingImports]
        QPixmap, QPainter, QFont, QColor, QPen, QBrush, QImage, QIcon, QPainterPath,
        QDragEnterEvent, QDropEvent, QRegion
    )
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    print("Warning: PySide6 not found. Please install it with 'pip install PySide6'")

from core.engine import ScanEngine, FileItem, Group


def _resolve_favicon_path() -> Optional[str]:
    """Locate favicon.ico for dev runs and PyInstaller onefile bundles."""
    candidates: List[str] = []
    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', '')
        if meipass:
            candidates.extend([
                os.path.join(meipass, 'icons', 'favicon.ico'),
                os.path.join(meipass, 'favicon.ico'),
            ])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join(script_dir, 'icons', 'favicon.ico'),
        os.path.join(script_dir, 'favicon.ico'),
        os.path.join(os.getcwd(), 'icons', 'favicon.ico'),
        os.path.join(os.getcwd(), 'favicon.ico'),
    ])
    seen: set = set()
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(norm):
            return norm
    return None


def _load_app_icon() -> Optional['QIcon']:
    """Load application icon from resolved favicon path."""
    if not PYSIDE6_AVAILABLE:
        return None
    icon_path = _resolve_favicon_path()
    if not icon_path:
        return None
    try:
        icon = QIcon(icon_path)
        return icon if not icon.isNull() else None
    except Exception:
        return None
from core.thumbnail_cache import PersistentThumbnailCache

if not PYSIDE6_AVAILABLE:
    print("Qt UI requires PySide6. Please install it with: pip install PySide6")
    sys.exit(1)



def format_byte_size(size_bytes: float) -> str:
    """Compact size label for summaries and group headers."""
    value = float(size_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


# Material Design 3 Dark Color Palette
MD3_COLORS = {
    'primary': "#D0BCFF",        # Light Purple/Lavender
    'on_primary': "#381E72",
    'primary_container': "#4F378B",
    'on_primary_container': "#EADDFF",
    'secondary': "#CCC2DC",
    'on_secondary': "#332D41",
    'surface': "#1C1B1F",        # Dark Surface
    'on_surface': "#E6E1E5",
    'surface_variant': "#49454F", # Darker Gray
    'on_surface_variant': "#CAC4D0",
    'outline': "#938F99",
    'error': "#F2B8B5",
    'on_error': "#601410",
    'error_container': "#8C1D18",
    'success': "#A5D6A7",        # Light Green
    'on_success': "#1B5E20",
    'bg_subtle': "#2B2930",      # Deep Gray
    'bg_tertiary': "#141218"     # Pure Dark
}


class ToggleSwitch(QWidget):
    """Material Design 3 style Toggle Switch."""
    toggled = Signal(bool)

    def __init__(self, parent=None, active_color="#D0BCFF", bg_color="#49454F"):
        super().__init__(parent)
        self.setFixedSize(52, 32)
        self._checked = False
        self._active_color = active_color
        self._bg_color = bg_color
        self._circle_pos = 4
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._animation.valueChanged.connect(self._update_pos)
        self.setStyleSheet("background-color: transparent;") # Ensure transparent background
        self.toggled.emit(self._checked)

    def _update_pos(self, value):
        self._circle_pos = value
        self.update()

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        if self._checked != value:
            self._checked = value
            start = self._circle_pos
            end = 24 if self._checked else 4
            self._animation.setStartValue(start)
            self._animation.setEndValue(end)
            self._animation.start()
            self.toggled.emit(self._checked)

    def mousePressEvent(self, event):
        self.setChecked(not self._checked)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Widget dimensions
        widget_height = self.height()  # 32px
        track_height = 24
        track_y = (widget_height - track_height) // 2  # Center the track vertically (4px from top)
        
        # Track
        color = QColor(self._active_color) if self._checked else QColor(self._bg_color)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, track_y, 52, track_height, 12, 12)
        
        # Thumb - center vertically within track
        thumb_color = QColor("#381E72") if self._checked else QColor("#CAC4D0")
        painter.setBrush(thumb_color)
        size = 16 if not self._checked else 20
        thumb_center_y = track_y + track_height // 2  # Center of track
        y_off = thumb_center_y - size // 2  # Center thumb vertically
        painter.drawEllipse(self._circle_pos, y_off, size, size)

class CustomTitleBar(QFrame):
    """Material Design 3 style custom title bar for frameless window."""
    def __init__(self, parent=None, title="CloneWiper"):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(48)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {MD3_COLORS['bg_tertiary']};
                border-bottom: 1px solid {MD3_COLORS['surface_variant']};
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 8, 0)
        layout.setSpacing(8)
        
        # Logo Icon (Favicon)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(28, 28)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("background-color: transparent; border: none;")
        
        # Load favicon for title bar logo
        icon_loaded = False
        app_icon = _load_app_icon()
        if app_icon is not None:
            try:
                pixmap = app_icon.pixmap(28, 28)
                if not pixmap.isNull():
                    self.icon_label.setPixmap(pixmap)
                    icon_loaded = True
            except Exception:
                pass
        
        if not icon_loaded:
            # Fallback to 'C' if favicon not found
            self.icon_label.setText("C")
            self.icon_label.setStyleSheet(f"""
                background-color: {MD3_COLORS['primary']};
                color: {MD3_COLORS['on_primary']};
                border-radius: 14px;
                font-weight: bold;
                font-size: 16px;
            """)
        layout.addWidget(self.icon_label)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"""
            color: {MD3_COLORS['on_surface']};
            font-size: 14px;
            font-weight: 500;
            font-family: 'Roboto', 'Segoe UI', sans-serif;
        """)
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Window Controls
        control_btn_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {MD3_COLORS['on_surface_variant']};
                border: none;
                width: 44px;
                height: 32px;
                font-size: 16px;
            }}
            QPushButton:hover {{ background-color: rgba(255, 255, 255, 0.1); }}
        """
        
        self.min_btn = QPushButton("—")
        self.min_btn.setStyleSheet(control_btn_style)
        self.min_btn.clicked.connect(self.parent.showMinimized)
        layout.addWidget(self.min_btn)
        
        self.max_btn = QPushButton("⬜")
        self.max_btn.setStyleSheet(control_btn_style)
        self.max_btn.clicked.connect(self._toggle_maximize)
        layout.addWidget(self.max_btn)
        
        self.close_btn = QPushButton("✕")
        self.close_btn.setStyleSheet(control_btn_style + "QPushButton:hover { background-color: #f44336; color: white; }")
        self.close_btn.clicked.connect(self.parent.close)
        layout.addWidget(self.close_btn)
        
        self._is_maximized = False
        self._dragging = False
        self._drag_pos = None

    def _toggle_maximize(self):
        if self._is_maximized:
            self.parent.showNormal()
        else:
            self.parent.showMaximized()
        self.set_maximized_state(not self._is_maximized)

    def set_maximized_state(self, is_maximized: bool):
        """Keep the custom title bar state in sync with OS-driven window changes."""
        self._is_maximized = is_maximized
        self.max_btn.setText("❐" if is_maximized else "⬜")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPos() - self.parent.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging and not self._is_maximized:
            self.parent.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize()
            event.accept()


class ThumbnailWorker(QRunnable):
    """Background worker for loading thumbnails."""
    PERSISTENT_CACHE_EXTS = {
        '.pdf', '.epub', '.mobi', '.azw3',
        '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v',
        '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus', '.alac', '.ape', '.aiff'
    }
    
    class Signals(QObject):
        thumb_ready = Signal(str, QPixmap, float)  # path, pixmap, aspect_ratio
        meta_ready = Signal(str, str)  # path, metadata_text
        error = Signal(str, str)  # path, error_msg
        
        def __init__(self, parent=None):
            super().__init__(parent)
    
    def __init__(self, file_path: str, max_width: int, page_token: float, cache: Optional[PersistentThumbnailCache] = None):
        super().__init__()
        self.file_path = file_path
        self.max_width = max_width
        self.page_token = page_token
        self.cache = cache
        # Create signals object - must be created in main thread for proper signal/slot connection
        # The signals object will be moved to main thread in _request_thumbnail
        self.signals = ThumbnailWorker.Signals()

    def _should_use_persistent_cache(self) -> bool:
        ext = os.path.splitext(self.file_path)[1].lower()
        return ext in self.PERSISTENT_CACHE_EXTS

    def _save_thumbnail_cache(self, qimg: QImage, width: int, height: int):
        """Persist the generated thumbnail so heavy media formats are fast next session."""
        if not self.cache or qimg.isNull() or not self._should_use_persistent_cache():
            return
        try:
            buf = QBuffer()
            buf.open(QBuffer.OpenModeFlag.ReadWrite)
            qimg.save(buf, "JPG", 85)
            data = buf.data().data()
            st = os.stat(self.file_path)
            mtime = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
            size = st.st_size
            self.cache.put(self.file_path, mtime, size, width, height, data)
        except Exception as e:
            logger.debug("Failed to cache %s: %s", self.file_path, e)
    
    def run(self):
        """Load thumbnail in background thread."""
        try:
            # logger.debug(f"DEBUG: ThumbnailWorker.run() called for {os.path.basename(self.file_path)}")
            ext = os.path.splitext(self.file_path)[1].lower()
            
            # --- PERSISTENT CACHE CHECK ---
            if self.cache and self._should_use_persistent_cache():
                try:
                    st = os.stat(self.file_path)
                    mtime = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
                    size = st.st_size
                    
                    cached_data = self.cache.get(self.file_path, mtime, size)
                    if cached_data:
                        # Cache hit!
                        qimg = QImage.fromData(cached_data)
                        if not qimg.isNull():
                            pixmap = QPixmap.fromImage(qimg)
                            aspect = 1.0
                            if qimg.height() > 0:
                                aspect = qimg.width() / qimg.height()
                            
                            self.signals.thumb_ready.emit(self.file_path, pixmap, aspect)
                            
                            # For metadata, we might still need to extract it if not cached, 
                            # but for now let's skip metadata on cache hit to be super fast,
                            # or we could cache metadata too. 
                            # Re-extracting metadata is usually fast enough or can be done lazily.
                            # For image dimensions, we interpret from the thumbnail but that might be 
                            # smaller than original.
                            # Ideally we store metadata in DB too.
                            # For now, let's re-emit basic metadata or "Cached"
                            # self.signals.meta_ready.emit(self.file_path, "") 
                            return
                except Exception as e:
                    logger.debug("Cache read error: %s", e)
            # -------------------------------

            # Image thumbnail (Standard formats)
            standard_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
            raw_exts = {
                '.cr2', '.crw', '.nef', '.nrw', '.arw', '.sr2', '.srf', '.srw', '.orf', '.raf',
                '.rw2', '.dng', '.raw', '.pef', '.ptx', '.rwl', '.3fr', '.ari', '.bay', '.cap',
                '.eip', '.iiq', '.cine', '.dcs', '.dcr', '.drf', '.erf', '.fff', '.mef', '.mos',
                '.mrw', '.r3d', '.rwz', '.x3f'
            }
            
            if ext in standard_exts:
                try:
                    from PIL import Image, ImageOps
                    with Image.open(self.file_path) as img:
                        img = ImageOps.exif_transpose(img)
                        # Use draft mode for JPEG
                        if img.format == 'JPEG' and hasattr(img, 'draft'):
                            img.draft('RGB', (self.max_width, self.max_width * 3))
                        if img.mode in ('RGBA', 'P'):
                            img = img.convert('RGB')
                        
                        width, height = img.size
                        if width <= 0 or height <= 0:
                            return
                        
                        aspect = width / height
                        base_width = min(self.max_width, 512)
                        scale = base_width / width
                        new_height = int(height * scale)
                        new_height = min(new_height, base_width * 3)
                        
                        img_resized = img.resize((base_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Convert to QPixmap
                        img_bytes = img_resized.tobytes('raw', 'RGB')
                        qimg = QImage(img_bytes, base_width, new_height, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        
                        self.signals.thumb_ready.emit(self.file_path, pixmap, aspect)
                        # logger.debug(f"DEBUG: ThumbnailWorker emitted thumb_ready for {os.path.basename(self.file_path)}")
                        
                        # Metadata
                        meta = f"{width}x{height}"
                        self.signals.meta_ready.emit(self.file_path, meta)
                except Exception as e:
                    logger.debug(f"DEBUG: ThumbnailWorker error for {os.path.basename(self.file_path)}: {e}")
                    self.signals.error.emit(self.file_path, str(e))
            
            # RAW Image thumbnail (rawpy)
            elif ext in raw_exts:
                try:
                    import rawpy
                    from PIL import Image
                    import io
                    
                    with rawpy.imread(self.file_path) as raw:
                        try:
                            # Try to extract embedded thumbnail first (fast)
                            thumb = raw.extract_thumb()
                            if thumb.format == rawpy.ThumbFormat.JPEG:
                                img = Image.open(io.BytesIO(thumb.data))
                            else:
                                # Bitmap thumbnail
                                img = Image.fromarray(thumb.data)
                        except Exception:
                            # Fallback to full postprocess (slow)
                            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, bright=1.0)
                            img = Image.fromarray(rgb)
                        
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            
                        width, height = img.size
                        aspect = width / height
                        base_width = min(self.max_width, 512)
                        scale = base_width / width
                        new_height = int(height * scale)
                        
                        img_resized = img.resize((base_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Convert to QPixmap
                        img_bytes = img_resized.tobytes('raw', 'RGB')
                        qimg = QImage(img_bytes, base_width, new_height, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        
                        self.signals.thumb_ready.emit(self.file_path, pixmap, aspect)
                        
                        # Metadata (Use full RAW size if possible)
                        raw_h, raw_w = raw.sizes.height, raw.sizes.width
                        meta = f"{raw_w}x{raw_h} (RAW)"
                        self.signals.meta_ready.emit(self.file_path, meta)

                except Exception as e:
                    logger.debug(f"DEBUG: ThumbnailWorker RAW error for {os.path.basename(self.file_path)}: {e}")
                    self.signals.error.emit(self.file_path, str(e))
            
            # Video thumbnail (OpenCV)
            elif ext in {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}:
                try:
                    try:
                        import cv2
                    except ImportError:
                        self.signals.error.emit(self.file_path, "OpenCV not available")
                        return
                    cap = cv2.VideoCapture(self.file_path)
                    if not cap.isOpened():
                        self.signals.error.emit(self.file_path, "Unable to open video")
                        return

                    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration_sec = int(frames / fps) if fps and fps > 0 else 0

                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        from PIL import Image
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        width, height = img.size
                        aspect = width / height
                        base_width = min(self.max_width, 512)
                        scale = base_width / width
                        new_height = int(height * scale)
                        img_resized = img.resize((base_width, new_height), Image.Resampling.LANCZOS)
                        img_bytes = img_resized.tobytes('raw', 'RGB')
                        qimg = QImage(img_bytes, base_width, new_height, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        self.signals.thumb_ready.emit(self.file_path, pixmap, aspect)
                        self._save_thumbnail_cache(qimg, video_w or width, video_h or height)
                        
                        # Metadata
                        dur_str = f"{duration_sec//60}:{duration_sec%60:02d}"
                        meta = f"{video_w or width}x{video_h or height} • {dur_str}"
                        self.signals.meta_ready.emit(self.file_path, meta)
                except Exception as e:
                    self.signals.error.emit(self.file_path, str(e))
            
            # PDF/EPUB/MOBI/AZW3 thumbnail (Robust Overhaul)
            elif ext in {'.pdf', '.epub', '.mobi', '.azw3'}:
                pdfium_error = None
                pymupdf_error = None
                
                try:
                    # PDF PRIMARY: pypdfium2 (High Fidelity)
                    if ext == '.pdf':
                        try:
                            import pypdfium2 as pdfium  # type: ignore[reportMissingImports]
                            from PIL import Image
                            import platform
                            
                            # Check if file exists and is readable
                            if not os.path.exists(self.file_path):
                                raise FileNotFoundError(f"PDF file not found: {self.file_path}")
                            if not os.access(self.file_path, os.R_OK):
                                raise PermissionError(f"PDF file not readable: {self.file_path}")
                            
                            pdf = pdfium.PdfDocument(self.file_path)
                            if len(pdf) == 0:
                                pdf.close()
                                raise ValueError("PDF has no pages")
                            
                            base_width = min(self.max_width, 512)
                            page = pdf[0]
                            try:
                                page_width, _page_height = page.get_size()
                                render_scale = max(0.25, min(2.0, base_width / page_width)) if page_width else 1.0
                            except Exception:
                                render_scale = 1.0
                            bitmap = page.render(scale=render_scale, rotation=0)
                            pil_image = bitmap.to_pil()
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            width, height = pil_image.size
                            aspect = width / height
                            scale = base_width / width
                            new_height = int(height * scale)
                            img_resized = pil_image.resize((base_width, new_height), Image.Resampling.LANCZOS)
                            img_bytes = img_resized.tobytes('raw', 'RGB')
                            qimg = QImage(img_bytes, base_width, new_height, QImage.Format.Format_RGB888)
                            pixmap = QPixmap.fromImage(qimg)
                            self.signals.thumb_ready.emit(self.file_path, pixmap, aspect)
                            self.signals.meta_ready.emit(self.file_path, f"{len(pdf)} pages")
                            self._save_thumbnail_cache(qimg, width, height)
                            pdf.close()
                            return
                        except ImportError as e:
                            pdfium_error = f"pypdfium2 not installed: {e}"
                            logger.debug(f"DEBUG: pypdfium2 import failed: {e}")
                        except Exception as e:
                            import traceback
                            pdfium_error = f"pypdfium2 error: {e}"
                            logger.debug(f"DEBUG: pypdfium2 processing failed for {os.path.basename(self.file_path)}: {e}")
                            logger.debug(f"DEBUG: Traceback: {traceback.format_exc()}")

                    # DOCUMENT FALLBACK/PRIMARY: PyMuPDF (fitz)
                    try:
                        import fitz  # type: ignore[reportMissingImports] # PyMuPDF
                        from PIL import Image
                        import platform
                        
                        # Check if file exists and is readable
                        if not os.path.exists(self.file_path):
                            raise FileNotFoundError(f"Document file not found: {self.file_path}")
                        if not os.access(self.file_path, os.R_OK):
                            raise PermissionError(f"Document file not readable: {self.file_path}")
                        
                        doc = fitz.open(self.file_path)
                        if doc.page_count == 0:
                            doc.close()
                            raise ValueError("Document has no pages")
                        
                        base_width = min(self.max_width, 512)
                        page = doc.load_page(0)
                        page_width = float(page.rect.width) if page.rect.width else base_width
                        zoom = max(0.25, min(2.0, base_width / page_width))
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        width, height = img.size
                        aspect = width / height
                        scale = base_width / width
                        new_height = int(height * scale)
                        img_resized = img.resize((base_width, new_height), Image.Resampling.LANCZOS)
                        img_bytes = img_resized.tobytes('raw', 'RGB')
                        qimg = QImage(img_bytes, base_width, new_height, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        self.signals.thumb_ready.emit(self.file_path, pixmap, aspect)
                        
                        meta = "Ebook"
                        if ext == '.pdf': meta = f"{doc.page_count} pages"
                        elif ext == '.epub': meta = "Ebook (EPUB)"
                        elif ext == '.mobi': meta = "Ebook (MOBI)"
                        elif ext == '.azw3': meta = "Ebook (AZW3)"
                        self.signals.meta_ready.emit(self.file_path, meta)
                        self._save_thumbnail_cache(qimg, width, height)
                        doc.close()
                        return
                    except ImportError as e:
                        pymupdf_error = f"PyMuPDF not installed: {e}"
                        logger.debug(f"DEBUG: PyMuPDF import failed: {e}")
                    except Exception as e:
                        import traceback
                        pymupdf_error = f"PyMuPDF error: {e}"
                        logger.debug(f"DEBUG: PyMuPDF processing failed for {os.path.basename(self.file_path)}: {e}")
                        logger.debug(f"DEBUG: Traceback: {traceback.format_exc()}")

                    # EPUB SPECIAL: Deep Manifest Parsing (Last Resort if fitz fails)
                    epub_fallback_success = False
                    if ext == '.epub':
                        import zipfile
                        import xml.etree.ElementTree as ET
                        from PIL import Image
                        cover_href = None  # Initialize to avoid UnboundLocalError
                        
                        try:
                            with zipfile.ZipFile(self.file_path) as z:
                                # 1. Find the .opf file
                                with z.open('META-INF/container.xml') as f:
                                    tree = ET.parse(f)
                                    opf_path = tree.find('.//{*}rootfile').get('full-path')
                                
                                # 2. Parse manifest for cover
                                opf_dir = os.path.dirname(opf_path)
                                with z.open(opf_path) as f:
                                    opf_tree = ET.parse(f)
                                    opf_root = opf_tree.getroot()
                                    ns = {'o': 'http://www.idpf.org/2007/opf', 'dc': 'http://purl.org/dc/elements/1.1/'}
                                    
                                    # Method A: item with properties="cover-image" (EPUB 3)
                                    for item in opf_root.findall('.//o:item[@properties="cover-image"]', ns):
                                        cover_href = item.get('href')
                                        if cover_href:
                                            break
                                    
                                    # Method B: meta name="cover" (EPUB 2) - only if Method A didn't find anything
                                    if not cover_href:
                                        meta_cover = opf_root.find('.//o:meta[@name="cover"]', ns)
                                        if meta_cover is not None:
                                            cover_id = meta_cover.get('content')
                                            if cover_id:
                                                item = opf_root.find(f'.//o:item[@id="{cover_id}"]', ns)
                                                if item is not None:
                                                    cover_href = item.get('href')

                                if cover_href:
                                    cover_full = os.path.join(opf_dir, cover_href).replace('\\', '/')
                                    with z.open(cover_full) as img_file:
                                        img = Image.open(img_file)
                                        if img.mode in ('RGBA', 'P'):
                                            img = img.convert('RGB')
                                        width, height = img.size
                                        aspect = width / height
                                        base_width = min(self.max_width, 512)
                                        scale = base_width / width
                                        new_height = int(height * scale)
                                        img_resized = img.resize((base_width, new_height), Image.Resampling.LANCZOS)
                                        img_bytes = img_resized.tobytes('raw', 'RGB')
                                        qimg = QImage(img_bytes, base_width, new_height, QImage.Format.Format_RGB888)
                                        pixmap = QPixmap.fromImage(qimg)
                                        self.signals.thumb_ready.emit(self.file_path, pixmap, aspect)
                                        self.signals.meta_ready.emit(self.file_path, "Ebook (EPUB-M)")
                                        self._save_thumbnail_cache(qimg, width, height)
                                        epub_fallback_success = True
                                        return
                        except Exception as epub_error:
                            # If EPUB special parsing fails, log but don't fail yet
                            logger.debug(f"DEBUG: EPUB fallback parsing failed for {os.path.basename(self.file_path)}: {epub_error}")
                            epub_fallback_success = False
                    
                    # If both PDF libraries failed, emit error with helpful message
                    if ext == '.pdf' and pdfium_error and pymupdf_error:
                        import platform
                        system = platform.system()
                        install_cmd = "pip3" if system == "Darwin" else "pip"
                        error_msg = f"PDF thumbnail failed. Both libraries failed:\n  pypdfium2: {pdfium_error}\n  PyMuPDF: {pymupdf_error}\n\nInstall one of them:\n  {install_cmd} install pypdfium2\n  or\n  {install_cmd} install PyMuPDF"
                        logger.debug(f"DEBUG: PDF thumbnail error for {os.path.basename(self.file_path)}: {error_msg}")
                        self.signals.error.emit(self.file_path, error_msg)
                    elif ext == '.epub' and pymupdf_error and not epub_fallback_success:
                        # EPUB: Only show error if both PyMuPDF and fallback parsing failed
                        import platform
                        system = platform.system()
                        install_cmd = "pip3" if system == "Darwin" else "pip"
                        error_msg = f"EPUB thumbnail failed.\n\nTo fix this, install PyMuPDF:\n  {install_cmd} install PyMuPDF\n\nError details: {pymupdf_error}\n\nNote: Some EPUB files may work without PyMuPDF if they contain cover images in the correct format."
                        logger.debug(f"DEBUG: EPUB thumbnail error for {os.path.basename(self.file_path)}: {error_msg}")
                        self.signals.error.emit(self.file_path, error_msg)
                    elif ext in {'.mobi', '.azw3'} and pymupdf_error:
                        # MOBI/AZW3: Require PyMuPDF (no fallback)
                        import platform
                        system = platform.system()
                        install_cmd = "pip3" if system == "Darwin" else "pip"
                        error_msg = f"Ebook thumbnail failed. Install PyMuPDF:\n  {install_cmd} install PyMuPDF\n\nError: {pymupdf_error}"
                        logger.debug(f"DEBUG: Ebook thumbnail error for {os.path.basename(self.file_path)}: {error_msg}")
                        self.signals.error.emit(self.file_path, error_msg)

                except Exception as e:
                    import traceback
                    error_msg = f"Doc error: {e}"
                    logger.debug(f"DEBUG: PDF/EPUB thumbnail error for {os.path.basename(self.file_path)}: {error_msg}")
                    logger.debug("%s", traceback.format_exc())
                    self.signals.error.emit(self.file_path, error_msg)
            
            # Audio thumbnail (Album Art via Mutagen)
            elif ext in {'.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus', '.alac', '.ape', '.aiff'}:
                try:
                    try:
                        import mutagen  # type: ignore[reportMissingImports]
                        from mutagen.mp3 import MP3  # type: ignore[reportMissingImports]
                        from mutagen.flac import FLAC  # type: ignore[reportMissingImports]
                        from mutagen.mp4 import MP4  # type: ignore[reportMissingImports]
                        from mutagen.oggvorbis import OggVorbis  # type: ignore[reportMissingImports]
                        from mutagen.oggopus import OggOpus  # type: ignore[reportMissingImports]
                    except ImportError:
                        # No thumbnail, but we can still emit some basic info if we want
                        # For now, just exit if mutagen isn't there
                        return
                    
                    audio = mutagen.File(self.file_path)
                    if audio is None:
                        return
                    
                    img_data = None
                    
                    # Extract album art for various formats
                    if ext == '.mp3' and audio.tags:
                        # MP3: Look for APIC frame
                        for tag in audio.tags.values():
                            if hasattr(tag, 'getID') and tag.getID() == 'APIC':
                                img_data = tag.data
                                break
                    elif ext == '.flac' and hasattr(audio, 'pictures') and audio.pictures:
                        # FLAC: Use pictures attribute
                        img_data = audio.pictures[0].data
                    elif ext in {'.m4a', '.aac', '.alac'} and audio.tags:
                        # MP4/M4A/AAC/ALAC: Look for 'covr' tag
                        if 'covr' in audio.tags:
                            img_data = audio.tags['covr'][0]
                    elif ext == '.ogg' and hasattr(audio, 'pictures') and audio.pictures:
                        # OGG Vorbis: Use pictures attribute
                        img_data = audio.pictures[0].data
                    elif ext == '.opus' and hasattr(audio, 'pictures') and audio.pictures:
                        # OGG Opus: Use pictures attribute
                        img_data = audio.pictures[0].data
                    elif ext == '.ape' and audio.tags:
                        # APE: Look for 'COVER ART (FRONT)' or 'Cover Art (Front)'
                        for key in audio.tags.keys():
                            if 'cover' in key.lower() or 'art' in key.lower():
                                img_data = audio.tags[key][0].value
                                break
                    
                    if img_data:
                        from PIL import Image
                        import io
                        try:
                            img = Image.open(io.BytesIO(img_data))
                            if img.mode in ('RGBA', 'P'):
                                img = img.convert('RGB')
                            width, height = img.size
                            aspect = width / height
                            base_width = min(self.max_width, 512)
                            scale = base_width / width
                            new_height = int(height * scale)
                            img_resized = img.resize((base_width, new_height), Image.Resampling.LANCZOS)
                            img_bytes = img_resized.tobytes('raw', 'RGB')
                            qimg = QImage(img_bytes, base_width, new_height, QImage.Format.Format_RGB888)
                            pixmap = QPixmap.fromImage(qimg)
                            self.signals.thumb_ready.emit(self.file_path, pixmap, aspect)
                            self._save_thumbnail_cache(qimg, width, height)
                        except Exception as img_error:
                            logger.debug(f"DEBUG: Failed to process album art image for {os.path.basename(self.file_path)}: {img_error}")
                    
                    # Metadata (Artist - Album, Duration)
                    try:
                        info = audio.info
                        duration = int(info.length) if info and hasattr(info, 'length') else 0
                        dur_str = f"{duration//60}:{duration%60:02d}" if duration > 0 else "0:00"
                    except Exception:
                        dur_str = "0:00"
                    
                    artist = "Unknown Artist"
                    album = "Unknown Album"
                    
                    if audio.tags:
                        try:
                            if ext == '.mp3':
                                # ID3v2 tags
                                artist_tag = audio.tags.get('TPE1') or audio.tags.get('TPE2')
                                album_tag = audio.tags.get('TALB')
                                if artist_tag:
                                    artist = str(artist_tag[0]) if isinstance(artist_tag, list) else str(artist_tag)
                                if album_tag:
                                    album = str(album_tag[0]) if isinstance(album_tag, list) else str(album_tag)
                            elif ext == '.flac':
                                # Vorbis comments
                                artist = ", ".join(audio.tags.get('artist', ['Unknown Artist']))
                                album = ", ".join(audio.tags.get('album', ['Unknown Album']))
                            elif ext in {'.m4a', '.aac', '.alac'}:
                                # MP4 tags
                                artist_tag = audio.tags.get('\xa9ART')
                                album_tag = audio.tags.get('\xa9alb')
                                if artist_tag:
                                    artist = ", ".join(artist_tag) if isinstance(artist_tag, list) else str(artist_tag)
                                if album_tag:
                                    album = ", ".join(album_tag) if isinstance(album_tag, list) else str(album_tag)
                            elif ext in {'.ogg', '.opus'}:
                                # Vorbis comments
                                artist = ", ".join(audio.tags.get('artist', ['Unknown Artist']))
                                album = ", ".join(audio.tags.get('album', ['Unknown Album']))
                            elif ext == '.ape':
                                # APE tags
                                artist_tag = audio.tags.get('Artist')
                                album_tag = audio.tags.get('Album')
                                if artist_tag:
                                    artist = str(artist_tag[0]) if isinstance(artist_tag, list) else str(artist_tag)
                                if album_tag:
                                    album = str(album_tag[0]) if isinstance(album_tag, list) else str(album_tag)
                        except Exception as tag_error:
                            logger.debug(f"DEBUG: Failed to extract tags for {os.path.basename(self.file_path)}: {tag_error}")
                    
                    meta = f"{artist} - {album} • {dur_str}"
                    self.signals.meta_ready.emit(self.file_path, meta)
                    
                except Exception as e:
                    logger.debug(f"DEBUG: Audio thumbnail error for {os.path.basename(self.file_path)}: {e}", exc_info=True)
                    self.signals.error.emit(self.file_path, f"Audio error: {e}")
        
        except Exception as e:
            self.signals.error.emit(self.file_path, str(e))


# One shared frame per file so copies in a group can be compared side by side.
COMPARE_CARD_WIDTH = 216
COMPARE_THUMB_HEIGHT = 136
COMPARE_CARD_HEIGHT = 224
COMPARE_STRIP_HEIGHT = COMPARE_CARD_HEIGHT + 28


class FileCardDelegate(QStyledItemDelegate):
    """Fixed-size card for a horizontal duplicate comparison strip."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thumbnails: Dict[str, QPixmap] = {}
        self.aspect_ratios: Dict[str, float] = {}
        self.metadata: Dict[str, str] = {}
        self.selection_state: Dict[str, bool] = {}
        self.card_width = 300
        self.mutex = QMutex()
        # Default aspect for files without thumbnails yet
        self._default_aspects: Dict[str, float] = {}
        # Text scrolling state for hover effect
        self._hovered_path: Optional[str] = None
        self._name_scroll_offset: Dict[str, float] = {}  # path -> scroll offset
        self._dir_scroll_offset: Dict[str, float] = {}  # path -> scroll offset
        self._scroll_timers: Dict[str, QTimer] = {}  # path -> timer
        self._text_widths: Dict[str, Tuple[float, float]] = {}  # path -> (name_width, dir_width)
    
    def set_card_width(self, width: int):
        self.card_width = max(200, min(width, 500))
    
    def set_thumbnail(self, path: str, pixmap: QPixmap, aspect: float):
        self.mutex.lock()
        try:
            self.thumbnails[path] = pixmap
            self.aspect_ratios[path] = aspect
            self._default_aspects.pop(path, None)
        finally:
            self.mutex.unlock()
    
    def set_metadata(self, path: str, meta: str):
        self.mutex.lock()
        try:
            self.metadata[path] = meta
        finally:
            self.mutex.unlock()
    
    def set_selection(self, path: str, selected: bool):
        self.mutex.lock()
        try:
            self.selection_state[path] = selected
        finally:
            self.mutex.unlock()
    
    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        """Every card uses the same frame so a group stays one comparison row."""
        return QSize(COMPARE_CARD_WIDTH, COMPARE_CARD_HEIGHT)
    
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Paint a fixed comparison card: same thumb window, then the facts that differ."""
        path = index.data(Qt.ItemDataRole.UserRole)
        if not path:
            return

        rect = option.rect
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.mutex.lock()
        try:
            selected = self.selection_state.get(path, False)
            thumb = self.thumbnails.get(path)
            meta = self.metadata.get(path, "")
        finally:
            self.mutex.unlock()

        border_radius = 12
        card_rect = rect.adjusted(1, 1, -1, -1)
        if selected:
            fill = QColor("#3A2426")
            border = QColor(MD3_COLORS['error'])
            border_width = 2
        elif path == self._hovered_path:
            fill = QColor(MD3_COLORS['surface'])
            border = QColor(MD3_COLORS['primary'])
            border_width = 2
        else:
            fill = QColor(MD3_COLORS['surface'])
            border = QColor(MD3_COLORS['surface_variant'])
            border_width = 1
        painter.setBrush(QBrush(fill))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(card_rect, border_radius, border_radius)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(border, border_width))
        painter.drawRoundedRect(card_rect, border_radius, border_radius)

        padding = 8
        thumb_rect = QRect(
            rect.x() + padding,
            rect.y() + padding,
            max(1, rect.width() - padding * 2),
            COMPARE_THUMB_HEIGHT,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(MD3_COLORS['surface_variant'])))
        painter.drawRoundedRect(thumb_rect, 8, 8)

        if thumb and not thumb.isNull():
            scaled = thumb.scaled(
                thumb_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            image_x = thumb_rect.x() + (thumb_rect.width() - scaled.width()) // 2
            image_y = thumb_rect.y() + (thumb_rect.height() - scaled.height()) // 2
            clip_path = QPainterPath()
            clip_path.addRoundedRect(thumb_rect, 8, 8)
            painter.setClipPath(clip_path)
            painter.drawPixmap(image_x, image_y, scaled)
            painter.setClipping(False)
        else:
            painter.setPen(QColor(MD3_COLORS['on_surface_variant']))
            placeholder_font = QFont("Roboto", 10)
            placeholder_font.setFamilies(["Roboto", "Segoe UI", "sans-serif"])
            painter.setFont(placeholder_font)
            painter.drawText(thumb_rect, Qt.AlignmentFlag.AlignCenter, "No preview")

        badge_label = "Delete" if selected else "Keep"
        badge = QRect(thumb_rect.x() + 8, thumb_rect.y() + 8, 58, 22)
        if selected:
            badge_fill = QColor(MD3_COLORS['error'])
            badge_text = QColor(MD3_COLORS['on_error'])
        else:
            badge_fill = QColor("#1B3D2F")
            badge_text = QColor(MD3_COLORS['success'])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(badge_fill))
        painter.drawRoundedRect(badge, 11, 11)
        painter.setPen(badge_text)
        badge_font = QFont("Roboto", 10, QFont.Weight.DemiBold)
        badge_font.setFamilies(["Roboto", "Segoe UI", "sans-serif"])
        painter.setFont(badge_font)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, badge_label)

        text_x = rect.x() + padding
        text_w = max(1, rect.width() - padding * 2)
        text_y = thumb_rect.bottom() + 8

        name = os.path.basename(path)
        try:
            size_str = self._format_size(os.path.getsize(path))
            mtime_str = time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(path)))
        except Exception:
            size_str = ""
            mtime_str = ""
        folder = os.path.basename(os.path.dirname(path)) or os.path.dirname(path)

        painter.setPen(QColor(MD3_COLORS['on_surface']))
        name_font = QFont("Roboto", 12, QFont.Weight.Medium)
        name_font.setFamilies(["Roboto", "Segoe UI", "sans-serif"])
        painter.setFont(name_font)
        name_rect = QRect(text_x, text_y, text_w, 18)
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, text_w),
        )

        detail = " · ".join(part for part in (size_str, meta, mtime_str) if part)
        painter.setPen(QColor(MD3_COLORS['on_surface_variant']))
        detail_font = QFont("Roboto", 10)
        detail_font.setFamilies(["Roboto", "Segoe UI", "sans-serif"])
        painter.setFont(detail_font)
        detail_rect = QRect(text_x, text_y + 20, text_w, 16)
        painter.drawText(
            detail_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(detail, Qt.TextElideMode.ElideRight, text_w),
        )
        folder_rect = QRect(text_x, text_y + 38, text_w, 16)
        painter.drawText(
            folder_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(folder, Qt.TextElideMode.ElideMiddle, text_w),
        )

        painter.restore()
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def set_hovered_path(self, path: Optional[str], view):
        """Repaint the card under the pointer so its border can show hover."""
        if path == self._hovered_path:
            return
        previous = self._hovered_path
        self._hovered_path = path
        model = view.model() if view is not None else None
        if model is None:
            return
        for candidate in (previous, path):
            if not candidate:
                continue
            try:
                for row in range(model.rowCount()):
                    index = model.index(row)
                    if index.data(Qt.ItemDataRole.UserRole) == candidate:
                        view.update(index)
                        break
            except (AttributeError, RuntimeError):
                return


class FileListModel(QAbstractListModel):
    """Model for file list in a group."""
    
    def __init__(self, files: List[str], parent=None):
        super().__init__(parent)
        self._files = files or []
    
    def rowCount(self, parent: QModelIndex = None) -> int:
        # For list models, return 0 if parent is valid (no children)
        if parent is None:
            parent = QModelIndex()
        # Check if parent is valid without calling isValid() to avoid recursion
        # In PySide6, an invalid QModelIndex has row() == -1 and column() == -1
        if parent.row() >= 0 or parent.column() >= 0:
            return 0
        return len(self._files)
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        path = self._files[index.row()]
        
        if role == Qt.ItemDataRole.UserRole:
            return path
        if role == Qt.ItemDataRole.ToolTipRole:
            return path
        
        return None
    
    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class GroupWidget(QFrame):
    """One duplicate group, shown as a fixed-height horizontal comparison strip."""
    
    selection_changed = Signal(str, bool) # path, selected
    
    def __init__(self, group_id: str, files: List[str], group_index: int = 0, total_groups: int = 0, parent=None):
        super().__init__(parent)
        self.group_id = group_id
        self.files = files
        self.is_expanded = True
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1) # Minimal padding to let internal view fill
        layout.setSpacing(0)
        
        # Header
        self.header = QFrame()
        self.header.setObjectName("groupHeader")
        self.header.setFixedHeight(48)
        self._update_header_style()
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(16, 4, 16, 4)
        header_layout.setSpacing(12)
        
        self.toggle_btn = QPushButton("▾")
        self.toggle_btn.setFixedSize(32, 32)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setToolTip("Collapse or expand this group")
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {MD3_COLORS['primary']};
                font-size: 16px;
                border: none;
                border-radius: 16px;
            }}
            QPushButton:hover {{ background-color: rgba(103, 80, 164, 0.1); }}
        """)
        self.toggle_btn.clicked.connect(self.toggle)
        header_layout.addWidget(self.toggle_btn)
        
        sizes = []
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    sizes.append(os.path.getsize(file_path))
                except OSError:
                    pass
        total_size = sum(sizes)
        reclaim = total_size - max(sizes) if len(sizes) > 1 else 0
        size_str = self._format_size(total_size)
        reclaim_str = self._format_size(reclaim)
        count_label = f"{len(files)} files"
        if total_groups > 0:
            title_text = f"Group {group_index + 1} of {total_groups}  ·  {count_label}  ·  {size_str}"
        else:
            title_text = f"{count_label}  ·  {size_str}"
        if reclaim > 0:
            title_text += f"  ·  {reclaim_str} to reclaim"
        title = QLabel(title_text)
        title.setStyleSheet(f"color: {MD3_COLORS['on_surface']}; background-color: transparent; font-size: 14px; font-weight: 500; font-family: 'Roboto', 'Segoe UI', sans-serif; border: none;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        layout.addWidget(self.header)
        
        # One horizontal row. Extra copies scroll sideways instead of wrapping
        # into a taller, uneven gallery.
        self.list_view = QListView()
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setSpacing(10)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setFlow(QListView.Flow.LeftToRight)
        self.list_view.setWrapping(False)
        self.list_view.setMovement(QListView.Movement.Static)
        self.list_view.setHorizontalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.list_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.list_view.setFrameShape(QFrame.Shape.NoFrame)
        self.list_view.setFixedHeight(COMPARE_STRIP_HEIGHT)
        self.list_view.setCursor(Qt.CursorShape.PointingHandCursor)
        self.list_view.setStyleSheet(f"""
            QListView {{
                background-color: transparent;
                border: none;
                padding: 8px 8px 0 8px;
            }}
            QScrollBar:horizontal {{
                background-color: transparent;
                height: 8px;
                margin: 0 8px 4px 8px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {MD3_COLORS['surface_variant']};
                border-radius: 4px;
                min-width: 24px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {MD3_COLORS['outline']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)
        
        self.delegate = FileCardDelegate()
        self.list_view.setItemDelegate(self.delegate)
        
        # Set model first before any operations that might trigger events
        self.model = FileListModel(files)
        self.list_view.setModel(self.model)
        
        # Auto-expand height
        self.model.layoutChanged.connect(self._update_height)
        self.model.dataChanged.connect(self._update_height)
        
        self.list_view.clicked.connect(self._on_item_clicked)
        
        # Add to layout first to ensure widget is properly initialized
        layout.addWidget(self.list_view)
        
        # Enable mouse tracking for hover effects after widget is in layout
        # This ensures the viewport is fully initialized
        self.list_view.setMouseTracking(True)
        if self.list_view.viewport():
            self.list_view.viewport().setMouseTracking(True)
            self.list_view.viewport().installEventFilter(self)
        
        # MD3 Card styling
        self.setStyleSheet(f"""
            GroupWidget {{
                background-color: {MD3_COLORS['surface']};
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 12px;
            }}
        """)
        
    def _update_height(self):
        """Keep the strip one card tall. Thumbnails no longer change the row height."""
        if not self.is_expanded:
            self.list_view.setFixedHeight(0)
            return
        if self.list_view.height() != COMPARE_STRIP_HEIGHT:
            self.list_view.setFixedHeight(COMPARE_STRIP_HEIGHT)
        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_height()
    
    def _format_size(self, size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def toggle(self):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.list_view.show()
            self.toggle_btn.setText("▾")
            self.list_view.setFixedHeight(COMPARE_STRIP_HEIGHT)
            self.list_view.doItemsLayout()
        else:
            self.list_view.hide()
            self.list_view.setFixedHeight(0)
            self.toggle_btn.setText("▸")
        self.updateGeometry()
        self._update_header_style()
    
    def _update_header_style(self):
        """Update header style with dynamic rounding based on expanded state."""
        bottom_radius = 0 if self.is_expanded else 11
        self.header.setStyleSheet(f"""
            QFrame#groupHeader {{
                background-color: {MD3_COLORS['bg_subtle']};
                border: none;
                border-top-left-radius: 11px;
                border-top-right-radius: 11px;
                border-bottom-left-radius: {bottom_radius}px;
                border-bottom-right-radius: {bottom_radius}px;
            }}
        """)

    def _on_item_entered(self, index: QModelIndex):
        """Handle mouse enter event for hover scrolling."""
        path = index.data(Qt.ItemDataRole.UserRole)
        if path:
            self.delegate.set_hovered_path(path, self.list_view)
    
    def eventFilter(self, obj, event):
        """Handle mouse move events for hover scrolling."""
        if obj == self.list_view.viewport():
            # Check if model is available before accessing it
            if self.list_view.model() is None:
                return super().eventFilter(obj, event)
            
            if event.type() == QEvent.Type.MouseMove:
                try:
                    index = self.list_view.indexAt(event.pos())
                    if index.isValid():
                        path = index.data(Qt.ItemDataRole.UserRole)
                        if path:
                            self.delegate.set_hovered_path(path, self.list_view)
                            return True
                except (AttributeError, RuntimeError):
                    # Model not fully initialized yet
                    return super().eventFilter(obj, event)
                # Mouse moved but not over any item
                self.delegate.set_hovered_path(None, self.list_view)
                return True
            elif event.type() == QEvent.Type.Leave:
                self.delegate.set_hovered_path(None, self.list_view)
                return True
        return super().eventFilter(obj, event)
    
    def _on_item_clicked(self, index: QModelIndex):
        path = index.data(Qt.ItemDataRole.UserRole)
        if path:
            current = self.delegate.selection_state.get(path, False)
            new_state = not current
            self.delegate.set_selection(path, new_state)
            self.list_view.update(index)
            self.selection_changed.emit(path, new_state)
    
    def set_selection(self, path: str, selected: bool):
        self.delegate.set_selection(path, selected)
        # Update view
        for i in range(self.model.rowCount()):
            idx = self.model.index(i)
            if idx.data(Qt.ItemDataRole.UserRole) == path:
                self.list_view.update(idx)
                break
    
    def set_thumbnail(self, path: str, pixmap: QPixmap, aspect: float):
        # Check if widget is still valid before updating
        try:
            if not hasattr(self, 'list_view') or self.list_view is None:
                return
            # Check if the C++ object is still alive
            _ = self.list_view.objectName()  # This will raise RuntimeError if deleted
        except RuntimeError:
            # Widget has been deleted, ignore the update
            return
        
        # Use QMetaObject.invokeMethod to ensure this runs on the main thread if called directly
        # though signals already handle this.
        self.delegate.set_thumbnail(path, pixmap, aspect)
        try:
            for i in range(self.model.rowCount()):
                idx = self.model.index(i)
                if idx.data(Qt.ItemDataRole.UserRole) == path:
                    self.list_view.update(idx)
                    break
        except RuntimeError:
            # Widget was deleted during update, ignore
            pass
    
    def set_metadata(self, path: str, meta: str):
        # Check if widget is still valid before updating
        try:
            if not hasattr(self, 'list_view') or self.list_view is None:
                return
            # Check if the C++ object is still alive
            _ = self.list_view.objectName()  # This will raise RuntimeError if deleted
        except RuntimeError:
            # Widget has been deleted, ignore the update
            return
        
        self.delegate.set_metadata(path, meta)
        # Update view
        try:
            for i in range(self.model.rowCount()):
                idx = self.model.index(i)
                if idx.data(Qt.ItemDataRole.UserRole) == path:
                    self.list_view.update(idx)
                    break
        except RuntimeError:
            # Widget was deleted during update, ignore
            pass


class CustomDialog(QDialog):
    """Custom Material Design 3 styled dialog."""
    
    def __init__(self, parent=None, title="", message="", buttons=None):
        super().__init__(parent)
        self.setWindowTitle("CloneWiper")
        # Cross-platform window flags (MSWindowsFixedSizeDialogHint is Windows-only)
        import platform
        flags = Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        if platform.system() == "Windows":
            flags |= Qt.WindowType.MSWindowsFixedSizeDialogHint
        self.setWindowFlags(flags)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        # Default buttons
        if buttons is None:
            buttons = ["OK"]
        
        self.setStyleSheet(f"""
            QDialog {{
                background-color: transparent;
            }}
            QFrame#dialogCard {{
                background-color: {MD3_COLORS['bg_subtle']};
                border: 2px solid {MD3_COLORS['outline']};
                border-radius: 12px;
            }}
        """)
        
        # Keep the native top-level dialog transparent; draw the visible rounded
        # surface in a child frame so square window corners do not show through.
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.dialog_card = QFrame(self)
        self.dialog_card.setObjectName("dialogCard")
        root_layout.addWidget(self.dialog_card)

        layout = QVBoxLayout(self.dialog_card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {MD3_COLORS['on_surface']};
                font-size: 18px;
                font-weight: 600;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                text-align: center;
            }}
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Message
        if message:
            message_label = QLabel(message)
            message_label.setStyleSheet(f"""
                QLabel {{
                    color: {MD3_COLORS['on_surface_variant']};
                    font-size: 13px;
                    font-family: 'Roboto', 'Segoe UI', sans-serif;
                    text-align: center;
                }}
            """)
            message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message_label.setWordWrap(True)
            # Enable HTML formatting
            message_label.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(message_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.addStretch()
        
        self.button_results = {}
        
        for i, button_text in enumerate(buttons):
            btn = QPushButton(button_text)
            btn.setFixedHeight(40)
            btn.setMinimumWidth(100)
            
            # Style buttons differently based on type
            if button_text.upper() in ["YES", "OK", "DELETE"]:
                # Primary/Error button
                if button_text.upper() == "DELETE":
                    bg_color = MD3_COLORS['error']
                    on_color = MD3_COLORS['on_error']
                    hover_color = MD3_COLORS['error']
                else:
                    bg_color = MD3_COLORS['primary']
                    on_color = MD3_COLORS['on_primary']
                    hover_color = MD3_COLORS['primary_container']
            else:
                # Secondary button (No, Cancel, etc.)
                bg_color = "transparent"
                on_color = MD3_COLORS['primary']
                hover_color = f"rgba(103, 80, 164, 0.08)"
            
            if button_text.upper() in ["YES", "OK", "DELETE"]:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {bg_color};
                        color: {on_color};
                        border: none;
                        border-radius: 20px;
                        padding: 8px 24px;
                        font-family: 'Roboto', 'Segoe UI', sans-serif;
                        font-size: 13px;
                        font-weight: 600;
                        min-width: 100px;
                    }}
                    QPushButton:hover {{
                        background-color: {hover_color};
                        opacity: 0.92;
                    }}
                    QPushButton:pressed {{
                        opacity: 0.8;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {bg_color};
                        color: {on_color};
                        border: 1px solid {MD3_COLORS['outline']};
                        border-radius: 20px;
                        padding: 8px 24px;
                        font-family: 'Roboto', 'Segoe UI', sans-serif;
                        font-size: 13px;
                        font-weight: 500;
                        min-width: 100px;
                    }}
                    QPushButton:hover {{
                        background-color: {hover_color};
                        border-color: {MD3_COLORS['primary']};
                    }}
                    QPushButton:pressed {{
                        background-color: rgba(103, 80, 164, 0.12);
                    }}
                """)
            
            btn.clicked.connect(lambda checked, text=button_text: self._on_button_clicked(text))
            button_layout.addWidget(btn)
            self.button_results[button_text] = btn
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Set default button
        if buttons:
            default_text = buttons[0] if len(buttons) == 1 else (buttons[1] if len(buttons) > 1 and buttons[1].upper() == "NO" else buttons[0])
            if default_text in self.button_results:
                self.button_results[default_text].setDefault(True)
                self.button_results[default_text].setFocus()
        
        self.result_text = None
    
    def _on_button_clicked(self, button_text: str):
        """Handle button click."""
        self.result_text = button_text
        self.accept()
    
    def showEvent(self, event):
        """Center dialog on parent window when shown."""
        super().showEvent(event)
        if self.parent():
            parent_rect = self.parent().geometry()
            dialog_rect = self.geometry()
            x = parent_rect.x() + (parent_rect.width() - dialog_rect.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - dialog_rect.height()) // 2
            self.move(x, y)
        else:
            # Center on screen if no parent
            app = QApplication.instance()
            if app and app.primaryScreen():
                screen = app.primaryScreen().geometry()
                dialog_rect = self.geometry()
                x = (screen.width() - dialog_rect.width()) // 2
                y = (screen.height() - dialog_rect.height()) // 2
                self.move(x, y)
    
    def exec(self) -> Optional[str]:
        """Execute dialog and return clicked button text."""
        super().exec()
        return self.result_text


class DragDropResultsWidget(QWidget):
    """Custom widget that supports drag and drop for folder selection."""
    
    folders_dropped = Signal(list)  # Emitted when folders are dropped (list of paths)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._drag_active = False
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            # Check if any URL is a directory
            urls = event.mimeData().urls()
            has_folder = any(url.isLocalFile() and os.path.isdir(url.toLocalFile()) for url in urls)
            if has_folder:
                event.acceptProposedAction()
                self._drag_active = True
                self.update()  # Trigger repaint for visual feedback
                return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self._drag_active = False
        self.update()  # Trigger repaint to remove visual feedback
        super().dragLeaveEvent(event)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event - supports multiple folders."""
        self._drag_active = False
        self.update()  # Trigger repaint to remove visual feedback
        
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            folders = []
            for url in urls:
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if os.path.isdir(path):
                        # Normalize path
                        path = os.path.normpath(path)
                        folders.append(path)
            
            if folders:
                # Emit all folders at once
                self.folders_dropped.emit(folders)
                event.acceptProposedAction()
                return
        event.ignore()
    
    def paintEvent(self, event):
        """Paint event to show drag and drop visual feedback."""
        super().paintEvent(event)
        
        if self._drag_active:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Draw dashed border to indicate drop zone
            pen = QPen(QColor(MD3_COLORS['primary']), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(MD3_COLORS['primary_container'] + '40')))  # Semi-transparent
            
            rect = self.rect().adjusted(10, 10, -10, -10)
            painter.drawRoundedRect(rect, 12, 12)
            
            # Draw text hint
            painter.setPen(QColor(MD3_COLORS['primary']))
            font = QFont("Roboto", 16, QFont.Weight.Medium)
            painter.setFont(font)
            text = "Drop folder(s) here to add to scan list"
            text_rect = painter.fontMetrics().boundingRect(text)
            text_x = (self.width() - text_rect.width()) // 2
            text_y = self.height() // 2
            painter.drawText(text_x, text_y, text)


class CloneWiperApp(QMainWindow):
    """Main CloneWiper application window."""
    
    # Signals for thread-safe UI updates
    progress_updated = Signal(float)
    status_updated = Signal(str)
    results_ready = Signal(dict)
    _is_windows = sys.platform.startswith("win")
    _resize_border = 8
    _win_ht_values = {
        "HTNOWHERE": 0,
        "HTCLIENT": 1,
        "HTCAPTION": 2,
        "HTLEFT": 10,
        "HTRIGHT": 11,
        "HTTOP": 12,
        "HTTOPLEFT": 13,
        "HTTOPRIGHT": 14,
        "HTBOTTOM": 15,
        "HTBOTTOMLEFT": 16,
        "HTBOTTOMRIGHT": 17,
        "HTMAXBUTTON": 9,
    }
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CloneWiper - Smart Duplicate Finder")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setGeometry(100, 100, 1400, 850)
        self.setMinimumSize(960, 640)
        
        # Set window icon (taskbar / Alt+Tab)
        app_icon = _load_app_icon()
        if app_icon is not None:
            self.setWindowIcon(app_icon)
        
        # Resize state
        self._resize_edge_active = False
        self._resize_start_pos = None
        self._resize_start_geom = None
        self.setMouseTracking(True)
        
        # Rounded corners radius
        self._corner_radius = 12
        
        # Connect signals to slots
        self.progress_updated.connect(self._on_progress_slot)
        self.status_updated.connect(self._on_status_slot)
        self.results_ready.connect(self._on_results_slot)
        
        # Engine
        self.engine = ScanEngine(
            progress_callback=self._on_progress,
            status_callback=self._on_status,
            results_callback=self._on_results
        )
        
        # State
        self.scan_paths: List[str] = []
        self.file_groups: Dict[str, List[str]] = {}
        self.file_groups_raw: Dict[str, List[str]] = {}
        self.selection_state: Dict[str, bool] = {}  # path -> should_delete
        self.current_page = 0
        self.groups_per_page = 100
        
        # Initialize file_groups to track scan state
        self.file_groups = None
        self.file_groups_raw = None

        # Persistent Thumbnail Cache
        self.persistent_cache = PersistentThumbnailCache()
        # VACUUM rewrites the whole cache file and can take seconds on a large cache;
        # run it in the background so the window isn't blocked from appearing.
        threading.Thread(target=lambda: self._cleanup_thumbnail_cache(vacuum=True), daemon=True).start()
        self._scroll_anchor: Optional[Tuple[str, int]] = None  # (group_id, scroll_value) for refresh restore
        
        # Thumbnail cache
        self.thumb_cache: Dict[str, QPixmap] = OrderedDict()
        self.thumb_cache_max = 1200
        self.thumb_aspects: Dict[str, float] = {}
        self.metadata_cache: Dict[str, str] = OrderedDict()
        self.metadata_cache_max = 4000
        self._group_widgets: Dict[str, GroupWidget] = {}  # group_id -> widget
        
        # Thumbnail loading tracking
        self._pending_thumbnails: Dict[str, bool] = {}  # path -> is_loading
        self._thumbnail_total = 0
        self._thumbnail_loaded = 0
        
        # Thread pool for thumbnails
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(16)
        
        # Page token for stale detection
        self._thumb_page_token = 0.0
        
        # Settings for remembering last browse path
        self.settings = QSettings("CloneWiper", "CloneWiper")
        
        self._setup_ui()
        
        # Don't auto-fill last browse path on app launch
    
    def _setup_ui(self):
        """Setup UI layout with Material Design 3."""
        # Define CenteredComboBox class early so it can be used throughout _setup_ui
        class CenteredComboBox(QComboBox):
            def paintEvent(self, event):
                # First, let the style draw the background and border
                opt = QStyleOptionComboBox()
                self.initStyleOption(opt)
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                
                # Draw the combo box frame (background and border)
                self.style().drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt, painter, self)
                
                # Now draw centered text over the background
                text_rect = self.rect()
                # Account for padding (16px on each side)
                text_rect.adjust(16, 0, -16, 0)
                
                painter.setPen(QColor(MD3_COLORS['on_surface']))
                # Match exact font settings from other buttons: 13px, weight 500 (Medium)
                # Use pixel size instead of point size to match CSS font-size exactly
                # Get font from the widget's style to ensure consistency
                font = self.font()
                font.setFamilies(["Roboto", "Segoe UI", "sans-serif"])
                font.setPixelSize(13)  # Use pixel size to match CSS exactly
                font.setWeight(QFont.Weight.Medium)  # 500
                painter.setFont(font)
                
                current_text = self.currentText()
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter, current_text)
        
        class CenteredComboDelegate(QStyledItemDelegate):
            def paint(self, painter, option, index):
                # Match exact font settings: 13px, weight 500 (Medium)
                font = QFont("Roboto", 13, QFont.Weight.Medium)
                font.setFamilies(["Roboto", "Segoe UI", "sans-serif"])
                painter.setFont(font)
                option.displayAlignment = Qt.AlignmentFlag.AlignCenter
                super().paint(painter, option, index)
        
        central = QWidget()
        self.setCentralWidget(central)
        central.setObjectName("centralWidget")
        # Set background color only to the central widget, not all children
        central.setStyleSheet(f"""
            QWidget#centralWidget {{
                background-color: {MD3_COLORS['surface']};
            }}
        """)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Custom Title Bar
        self.custom_title_bar = CustomTitleBar(self)
        self.main_layout.addWidget(self.custom_title_bar)
        
        # Control panel (Material 3 Card) - Condensed to single row
        control_panel = QFrame()
        control_panel.setMinimumHeight(48) # Minimum height, can expand for folder list
        control_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {MD3_COLORS['bg_subtle']};
                border-bottom: 1px solid {MD3_COLORS['surface_variant']};
            }}
        """)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 12, 20, 12)
        control_layout.setSpacing(12)
        
        # Path selection area (Condensed)
        path_container = QFrame()
        path_container.setStyleSheet("background-color: transparent;")
        path_hbox = QHBoxLayout(path_container)
        path_hbox.setContentsMargins(0, 0, 0, 0)
        path_hbox.setSpacing(12)
        
        path_label = QLabel("Folders")
        path_label.setStyleSheet(f"color: {MD3_COLORS['on_surface_variant']}; font-size: 13px; font-weight: 600; font-family: 'Roboto', 'Segoe UI', sans-serif;")
        path_hbox.addWidget(path_label)
        
        # Path List Widget (With Scrollbar)
        self.path_list_widget = QListWidget()
        self.path_list_widget.setMinimumWidth(300)
        self.path_list_widget.setFixedHeight(36)
        self.path_list_widget.setToolTip("Folders included in the next scan. Select one and press Delete to remove it.")
        self.path_list_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.path_list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.path_list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {MD3_COLORS['bg_subtle']};
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 8px;
                color: {MD3_COLORS['on_surface']};
                font-size: 12px;
                outline: none;
                padding: 0px;
            }}
            QListWidget::item {{
                padding: 4px 8px;
                border-radius: 6px;
                margin: 2px;
                min-height: 28px;
            }}
            QListWidget::item:selected {{
                background-color: {MD3_COLORS['primary_container']};
                color: {MD3_COLORS['on_primary_container']};
            }}
            QScrollBar:vertical {{
                background-color: transparent;
                width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {MD3_COLORS['surface_variant']};
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        # Auto-grow path list height
        self.path_list_widget.model().rowsInserted.connect(self._adjust_path_list_height)
        self.path_list_widget.model().rowsRemoved.connect(self._adjust_path_list_height)
        
        # Folder removal via row action, context menu, Delete, and Backspace.
        self.path_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.path_list_widget.customContextMenuRequested.connect(self._show_path_context_menu)
        self.path_list_widget.keyPressEvent = self._path_list_key_press
        
        path_hbox.addWidget(self.path_list_widget)
        
        # Add Folder Button (Icon Only or Small)
        add_btn = QPushButton("Add folder")
        add_btn.setFixedHeight(36)
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.setToolTip("Choose a folder to scan")
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {MD3_COLORS['primary_container']};
                color: {MD3_COLORS['on_primary_container']};
                font-size: 13px;
                font-weight: 600;
                border: none;
                border-radius: 18px;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background-color: #EADDFF;
                color: #381E72;
            }}
            QPushButton:focus {{
                border: 2px solid {MD3_COLORS['primary']};
            }}
        """)
        add_btn.clicked.connect(self._browse_path)
        path_hbox.addWidget(add_btn)
        
        control_layout.addWidget(path_container)
        
        # Add stretch to push controls to the right
        control_layout.addStretch()
        
        # Hash Mode Selection (no label, centered text like other dropdowns)
        self.hash_mode_combo = QComboBox()
        self.hash_mode_combo.addItems([
            "Exact match",
            "Similar photos",
            "Similar photos (recommended)",
            "Similar + visual check",
            "Strict + visual check",
            "Strict + AI check",
        ])
        self._hash_mode_tips = [
            "MD5. Keeps only byte-for-byte identical files. Fastest, and the right choice for documents and audio.",
            "One perceptual hash. Also finds resized or recompressed photos, with more lookalikes.",
            "Four perceptual hashes have to mostly agree. Recommended balance of speed and accuracy.",
            "Perceptual hash, then an ORB feature check, so fewer lookalikes survive.",
            "Four-hash vote plus an ORB check. Strictest visual mode, and the slowest.",
            "Four-hash vote plus an AI embedding check. Use when a false match would be costly.",
        ]
        self.hash_mode_combo.setCurrentIndex(2)  # Default to multi-algorithm
        self.hash_mode_combo.setFixedHeight(36)
        self.hash_mode_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hash_mode_combo.setToolTip(self._hash_mode_tips[2])
        for index, tip in enumerate(self._hash_mode_tips):
            self.hash_mode_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        self.hash_mode_combo.currentIndexChanged.connect(
            lambda index: self.hash_mode_combo.setToolTip(self._hash_mode_tips[index])
        )
        self.hash_mode_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {MD3_COLORS['bg_subtle']};
                color: {MD3_COLORS['on_surface']};
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 8px;
                min-width: 240px;
                padding: 4px 28px 4px 12px;
                font-size: 13px;
                font-weight: 600;
            }}
            QComboBox:hover {{
                border-color: {MD3_COLORS['primary']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {MD3_COLORS['bg_subtle']};
                color: {MD3_COLORS['on_surface']};
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 8px;
                selection-background-color: {MD3_COLORS['primary_container']};
                selection-color: {MD3_COLORS['on_primary_container']};
                font-size: 13px;
                font-weight: 500;
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 12px;
                border-radius: 4px;
            }}
        """)
        control_layout.addWidget(self.hash_mode_combo)
        
        # Scan Button (Filled)
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.setFixedHeight(36)
        self.scan_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scan_btn.setToolTip("Scan the folders above for duplicates")
        self.scan_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {MD3_COLORS['primary']};
                color: {MD3_COLORS['on_primary']};
                font-size: 13px;
                font-weight: 700;
                border: none;
                border-radius: 18px;
                padding: 0 22px;
            }}
            QPushButton:hover {{
                background-color: #EADDFF;
                color: #381E72;
            }}
            QPushButton:focus {{
                border: 2px solid {MD3_COLORS['on_primary']};
            }}
            QPushButton:disabled {{
                background-color: {MD3_COLORS['surface_variant']};
                color: {MD3_COLORS['on_surface_variant']};
            }}
        """)
        self.scan_btn.clicked.connect(self._start_scanning)
        control_layout.addWidget(self.scan_btn)
        self.main_layout.addWidget(control_panel)
        
        # Status bar (Material 3)
        status_bar = QFrame()
        self.status_bar = status_bar
        status_bar.setFixedHeight(36)
        status_bar.setVisible(False)
        status_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {MD3_COLORS['bg_tertiary']};
                border: none;
            }}
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(24, 0, 24, 0)
        status_layout.setSpacing(16)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"""
            color: {MD3_COLORS['on_surface']};
            font-size: 13px;
            font-weight: 500;
            font-family: 'Roboto', 'Segoe UI', sans-serif;
            padding: 0px;
        """)
        status_layout.addWidget(self.status_label)
        # Hide status label initially (will show during scanning/results)
        self.status_label.setVisible(False)
        
        status_layout.addStretch()
        
        # Progress bar removed from status bar (now in center of results area)
        
        self.main_layout.addWidget(status_bar)
        
        # Results area (scrollable) - Material 3
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {MD3_COLORS['bg_tertiary']};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: transparent;
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {MD3_COLORS['surface_variant']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {MD3_COLORS['outline']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        
        # Create drag-and-drop enabled results container
        self.results_container = DragDropResultsWidget()
        self.results_container.setStyleSheet(f"background-color: {MD3_COLORS['bg_tertiary']};")
        self.results_container.folders_dropped.connect(self._on_folders_dropped)
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(24, 16, 24, 16)
        self.results_layout.setSpacing(16)
        
        # Center container for scanning progress and initial instruction
        self.center_progress_container = QWidget()
        center_layout = QVBoxLayout(self.center_progress_container)
        center_layout.setContentsMargins(24, 24, 24, 24)
        center_layout.setSpacing(16)
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Center status label for scanning progress and initial instruction
        self.center_status_label = QLabel()
        self.center_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Enable word wrapping for long status messages
        self.center_status_label.setWordWrap(True)
        # Enable HTML formatting
        self.center_status_label.setTextFormat(Qt.TextFormat.RichText)
        # Set width constraints to match progress bar and prevent overflow
        self.center_status_label.setMinimumWidth(600)
        self.center_status_label.setMaximumWidth(800)
        self.center_status_label.setStyleSheet(f"""
            QLabel {{
                color: {MD3_COLORS['on_surface']};
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px;
            }}
        """)
        center_layout.addWidget(self.center_status_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Set initial instruction text
        self.center_status_label.setText(self._get_instruction_text())
        
        # Center progress bar (below status label, initially hidden)
        self.center_progress_bar = QProgressBar()
        self.center_progress_bar.setFixedHeight(8)
        self.center_progress_bar.setMinimumWidth(600)
        self.center_progress_bar.setMaximumWidth(800)
        self.center_progress_bar.setTextVisible(False)
        self.center_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {MD3_COLORS['surface_variant']};
                border: none;
                border-radius: 4px;
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {MD3_COLORS['primary']};
                border-radius: 4px;
            }}
        """)
        center_layout.addWidget(self.center_progress_bar, 0, Qt.AlignmentFlag.AlignCenter)
        # Initially show the instruction (center_progress_container visible, progress bar hidden)
        self.center_progress_container.setVisible(True)
        self.center_progress_bar.setVisible(False)
        
        # Add center container to results layout with stretch for vertical centering
        self.results_layout.addStretch()
        self.results_layout.addWidget(self.center_progress_container, 0, Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addStretch()
        
        self.scroll_area.setWidget(self.results_container)
        self.main_layout.addWidget(self.scroll_area)
        
        # KEY: Apply stretch factors to prevent top widgets from vertical expansion
        self.main_layout.setStretch(0, 0) # Custom Title Bar
        self.main_layout.setStretch(1, 0) # Control Panel
        self.main_layout.setStretch(2, 0) # Status Bar
        self.main_layout.setStretch(3, 1) # Results Area (Grow)
        
        # Footer (Material 3 Bottom App Bar style) - Wrap in Scroll Area
        self.footer = QScrollArea()
        self.footer.setFixedHeight(64)
        self.footer.setWidgetResizable(True)
        self.footer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.footer.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.footer.setFrameShape(QFrame.Shape.NoFrame)
        # Initially hide footer (no results)
        self.footer.setVisible(False)
        self.footer.setStyleSheet(f"""
            QScrollArea {{
                background-color: {MD3_COLORS['bg_tertiary']};
                border: none;
                border-top: 1px solid {MD3_COLORS['surface_variant']};
            }}
            QScrollBar:horizontal {{
                background-color: transparent;
                height: 4px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {MD3_COLORS['surface_variant']};
                border-radius: 2px;
                min-width: 20px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)

        footer_content = QWidget()
        footer_content.setObjectName("footerContent")
        footer_content.setStyleSheet(f"QWidget#footerContent {{ background-color: {MD3_COLORS['bg_tertiary']}; border: none; }}")
        footer_layout = QHBoxLayout(footer_content)
        footer_layout.setContentsMargins(12, 0, 12, 0)
        footer_layout.setSpacing(6)
        footer_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)  # Vertical center alignment
        
        # Quick select buttons (Material 3 Outlined Button)
        outlined_button_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {MD3_COLORS['primary']};
                font-size: 13px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 12px;
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 20px;
                min-width: 80px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: rgba(103, 80, 164, 0.08);
                border-color: {MD3_COLORS['primary']};
            }}
            QPushButton:pressed {{
                background-color: rgba(103, 80, 164, 0.12);
            }}
            QPushButton[quickSelected="true"] {{
                background-color: {MD3_COLORS['primary_container']};
                color: {MD3_COLORS['on_primary_container']};
                border-color: {MD3_COLORS['primary']};
            }}
            QPushButton[quickSelected="true"]:hover {{
                background-color: {MD3_COLORS['primary_container']};
                color: {MD3_COLORS['on_primary_container']};
            }}
        """
        
        self.keep_newest_btn = QPushButton("Keep newest")
        self.keep_newest_btn.setFixedHeight(40)
        self.keep_newest_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.keep_newest_btn.setToolTip("Keep the most recently modified file in each group")
        self.keep_newest_btn.setStyleSheet(outlined_button_style)
        self.keep_newest_btn.clicked.connect(lambda: self._quick_select('newest'))
        footer_layout.addWidget(self.keep_newest_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.keep_oldest_btn = QPushButton("Keep oldest")
        self.keep_oldest_btn.setFixedHeight(40)
        self.keep_oldest_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.keep_oldest_btn.setToolTip("Keep the earliest file in each group")
        self.keep_oldest_btn.setStyleSheet(outlined_button_style)
        self.keep_oldest_btn.clicked.connect(lambda: self._quick_select('oldest'))
        footer_layout.addWidget(self.keep_oldest_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.keep_best_btn = QPushButton("Keep best")
        self.keep_best_btn.setFixedHeight(40)
        self.keep_best_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.keep_best_btn.setToolTip("Keep the highest-resolution image, then the largest file")
        self.keep_best_btn.setStyleSheet(outlined_button_style)
        self.keep_best_btn.clicked.connect(lambda: self._quick_select('best_res'))
        footer_layout.addWidget(self.keep_best_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.keep_smallest_btn = QPushButton("Keep smallest")
        self.keep_smallest_btn.setFixedHeight(40)
        self.keep_smallest_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.keep_smallest_btn.setToolTip("Keep the highest-resolution image, then the smallest file")
        self.keep_smallest_btn.setStyleSheet(outlined_button_style)
        self.keep_smallest_btn.clicked.connect(lambda: self._quick_select('keep_smallest'))
        footer_layout.addWidget(self.keep_smallest_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.keep_raw_btn = QPushButton("Keep RAW")
        self.keep_raw_btn.setFixedHeight(40)
        self.keep_raw_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.keep_raw_btn.setToolTip("Prefer the RAW file when a group also contains a JPEG")
        self.keep_raw_btn.setStyleSheet(outlined_button_style)
        self.keep_raw_btn.clicked.connect(lambda: self._quick_select('keep_raw'))
        footer_layout.addWidget(self.keep_raw_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.quick_select_buttons = {
            'newest': self.keep_newest_btn,
            'oldest': self.keep_oldest_btn,
            'best_res': self.keep_best_btn,
            'keep_smallest': self.keep_smallest_btn,
            'keep_raw': self.keep_raw_btn,
        }
        
        # Group controls on the left
        footer_layout.addSpacing(8)
        
        # Scope toggle (Switch)
        toggle_container = QWidget()
        toggle_hbox = QHBoxLayout(toggle_container)
        toggle_hbox.setContentsMargins(0, 0, 0, 0)
        toggle_hbox.setSpacing(8)
        toggle_hbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)  # Vertical center alignment
        toggle_container.setStyleSheet("background-color: transparent;") # Ensure transparent background for container
        
        self.scope_label = QLabel("This page")
        self.scope_label.setMinimumWidth(72)
        self.scope_label.setStyleSheet(f"color: {MD3_COLORS['on_surface']}; font-size: 13px; font-family: 'Roboto', 'Segoe UI', sans-serif;")
        self.scope_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        
        self.scope_toggle = ToggleSwitch(active_color=MD3_COLORS['primary'], bg_color=MD3_COLORS['surface_variant'])
        self.scope_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scope_toggle.setToolTip("Apply Keep actions to this page, or to every page")
        self.scope_toggle.toggled.connect(self._update_scope_label)
        
        toggle_hbox.addWidget(self.scope_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        toggle_hbox.addWidget(self.scope_toggle, alignment=Qt.AlignmentFlag.AlignVCenter)
        footer_layout.addWidget(toggle_container, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        footer_layout.addSpacing(8)
        
        # Sort (No Label) - Custom QComboBox with centered text
        # Note: CenteredComboBox is already defined at the start of _setup_ui
        class PageComboBox(CenteredComboBox):
            """ComboBox for page selection that displays 'current/total' in the button but only page numbers in dropdown."""
            def __init__(self, parent=None):
                super().__init__(parent)
                self._total_pages = 1
            
            def set_total_pages(self, total: int):
                """Set the total number of pages for display."""
                self._total_pages = total
                self.update()  # Trigger repaint
            
            def paintEvent(self, event):
                # First, let the style draw the background and border
                opt = QStyleOptionComboBox()
                self.initStyleOption(opt)
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                
                # Draw the combo box frame (background and border)
                self.style().drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt, painter, self)
                
                # Now draw centered text over the background
                text_rect = self.rect()
                # Account for padding (16px on each side)
                text_rect.adjust(16, 0, -16, 0)
                
                painter.setPen(QColor(MD3_COLORS['on_surface']))
                # Match exact font settings: 14px, weight 500 (Medium)
                font = self.font()
                font.setFamilies(["Roboto", "Segoe UI", "sans-serif"])
                font.setPixelSize(14)  # Match page_combo font size
                font.setWeight(QFont.Weight.Medium)  # 500
                painter.setFont(font)
                
                # Display "current/total" format in the button
                current_page = self.currentIndex() + 1 if self.currentIndex() >= 0 else 1
                display_text = f"{current_page}/{self._total_pages}"
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter, display_text)
        
        self.sort_combo = CenteredComboBox()
        self.sort_combo.setFixedHeight(40)
        self.sort_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: transparent;
                color: {MD3_COLORS['on_surface']};
                font-size: 13px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 16px;
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 20px;
                min-width: 200px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 0px;
                background-color: transparent;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                padding: 0px;
                margin: 0px;
            }}
            QComboBox:hover {{
                border-color: {MD3_COLORS['on_surface']};
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {MD3_COLORS['surface']};
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 8px;
                selection-background-color: {MD3_COLORS['primary_container']};
                selection-color: {MD3_COLORS['on_primary_container']};
                outline: none;
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 16px;
                border-radius: 4px;
                min-height: 20px;
                text-align: center;
                font-size: 13px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {MD3_COLORS['surface_variant']};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {MD3_COLORS['primary_container']};
                color: {MD3_COLORS['on_primary_container']};
            }}
        """)
        sort_items = [
            "Count (High-Low)", "Count (Low-High)", 
            "File Size (Large-Small)", "File Size (Small-Large)",
            "Name (A-Z)", "Name (Z-A)", "Newest First", "Oldest First"
        ]
        self.sort_combo.addItems(sort_items)
        
        # Calculate minimum width based on longest text
        font_metrics = self.sort_combo.fontMetrics()
        max_width = max(font_metrics.horizontalAdvance(item) for item in sort_items)
        # Add padding (16px on each side) plus some extra space
        min_width = max_width + 32 + 20  # 32px for padding, 20px extra buffer
        self.sort_combo.setMinimumWidth(min_width)
        
        # Set text alignment to center for dropdown items
        # Note: CenteredComboDelegate is already defined at the start of _setup_ui
        self.sort_combo.setItemDelegate(CenteredComboDelegate(self.sort_combo))
        self.sort_combo.setToolTip("Sort duplicate groups")
        self.sort_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sort_combo.currentTextChanged.connect(self._apply_sorting)
        footer_layout.addWidget(self.sort_combo, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        footer_layout.addSpacing(8)
        
        # Paging (Material 3 Icon Buttons)
        icon_button_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {MD3_COLORS['primary']};
                font-size: 24px;
                font-weight: 500;
                border: none;
                border-radius: 20px;
                min-width: 40px;
                min-height: 40px;
            }}
            QPushButton:hover {{
                background-color: {MD3_COLORS['primary_container']};
            }}
            QPushButton:pressed {{
                background-color: {MD3_COLORS['primary_container']};
                opacity: 0.8;
            }}
            QPushButton:disabled {{
                color: {MD3_COLORS['on_surface_variant']};
                opacity: 0.38;
            }}
        """
        
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedSize(40, 40)
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.setToolTip("Previous page")
        self.prev_btn.setStyleSheet(icon_button_style)
        self.prev_btn.clicked.connect(self._prev_page)
        footer_layout.addWidget(self.prev_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        # Page selector combo box (clickable page indicator)
        self.page_combo = PageComboBox()
        self.page_combo.setFixedSize(90, 40)
        self.page_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: transparent;
                color: {MD3_COLORS['on_surface']};
                font-size: 14px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 8px;
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 20px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
                background-color: transparent;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                padding: 0px 4px;
            }}
            QComboBox:hover {{
                border-color: {MD3_COLORS['on_surface']};
                background-color: {MD3_COLORS['surface_variant']};
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {MD3_COLORS['surface']};
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 8px;
                selection-background-color: {MD3_COLORS['primary_container']};
                selection-color: {MD3_COLORS['on_primary_container']};
                outline: none;
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 16px;
                border-radius: 4px;
                min-height: 20px;
                text-align: center;
                font-size: 14px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {MD3_COLORS['surface_variant']};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {MD3_COLORS['primary_container']};
                color: {MD3_COLORS['on_primary_container']};
            }}
        """)
        self.page_combo.setItemDelegate(CenteredComboDelegate(self.page_combo))
        self.page_combo.currentIndexChanged.connect(self._on_page_selected)
        footer_layout.addWidget(self.page_combo, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedSize(40, 40)
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.setToolTip("Next page")
        self.next_btn.setStyleSheet(icon_button_style)
        self.next_btn.clicked.connect(self._next_page)
        footer_layout.addWidget(self.next_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        footer_layout.addStretch()
        
        # Delete button (Material 3 Error Button)
        self.delete_btn = QPushButton("Delete selected")
        self.delete_btn.setFixedHeight(40)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setToolTip("Move the marked copies to the Recycle Bin. One file in each group should stay.")
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {MD3_COLORS['error']};
                color: {MD3_COLORS['on_error']};
                font-size: 14px;
                font-weight: 600;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 10px 24px;
                border: none;
                border-radius: 20px;
                min-width: 160px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {MD3_COLORS['error']};
                opacity: 0.92;
            }}
            QPushButton:pressed {{
                background-color: {MD3_COLORS['error']};
                opacity: 0.8;
            }}
            QPushButton:disabled {{
                background-color: {MD3_COLORS['surface_variant']};
                color: {MD3_COLORS['on_surface_variant']};
                opacity: 0.38;
            }}
        """)
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_selected)
        footer_layout.addWidget(self.delete_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        # Clear Selection button (Material 3 Text/Outlined Button)
        self.clear_select_btn = QPushButton("Clear")
        self.clear_select_btn.setFixedHeight(40)
        self.clear_select_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_select_btn.setToolTip("Unmark every file currently marked for deletion")
        self.clear_select_btn.setStyleSheet(outlined_button_style)
        self.clear_select_btn.setEnabled(False)
        self.clear_select_btn.clicked.connect(self._clear_selection)
        footer_layout.addWidget(self.clear_select_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.footer.setWidget(footer_content)
        self.main_layout.addWidget(self.footer)
    
    def _get_instruction_text(self) -> str:
        """Empty-state copy. Mode details live on the mode menu tooltip."""
        return (
            "<div style='text-align: center;'>"
            "<p style='font-size: 28px; font-weight: 600; margin: 0 0 8px 0;'>"
            "Find duplicate files"
            "</p>"
            "<p style='font-size: 15px; color: #CAC4D0; margin: 0 0 16px 0;'>"
            "Add a folder, or drop one anywhere in this area."
            "</p>"
            "<p style='font-size: 13px; color: #938F99; margin: 0; line-height: 1.55;'>"
            "Exact match finds identical files.<br>"
            "Similar photos also catches resized and recompressed images.<br>"
            "Visual and AI checks drop lookalikes before anything is deleted."
            "</p>"
            "</div>"
        )
    
    def _adjust_path_list_height(self):
        """Dynamically adjust path list height based on items (max 1.5 rows)."""
        count = self.path_list_widget.count()
        if count <= 1:
            h = 36
            if count == 1:
                item = self.path_list_widget.item(0)
                if item:
                    item.setSizeHint(QSize(-1, 32))
        else:
            h = 58
            for i in range(count):
                item = self.path_list_widget.item(i)
                if item:
                    item.setSizeHint(QSize(-1, 32))
        self.path_list_widget.setFixedHeight(h)
        self.updateGeometry()

    def _path_list_key_press(self, event):
        """Handle Delete/Backspace key for folder removal."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._remove_selected_folder()
        else:
            # Fallback to standard QListWidget key handling
            QListWidget.keyPressEvent(self.path_list_widget, event)

    def _remove_folder_item(self, item: Optional[QListWidgetItem]):
        """Remove a folder row from the scan list."""
        if not item:
            return
        row = self.path_list_widget.row(item)
        if row >= 0:
            removed = self.path_list_widget.takeItem(row)
            if removed:
                del removed

    def _remove_selected_folder(self):
        """Remove the currently selected folder row."""
        self._remove_folder_item(self.path_list_widget.currentItem())

    def _show_path_context_menu(self, position):
        from PySide6.QtWidgets import QMenu  # type: ignore[reportMissingImports]
        menu = QMenu()
        remove_action = menu.addAction("Remove Folder")
        action = menu.exec(self.path_list_widget.mapToGlobal(position))
        if action == remove_action:
            self._remove_selected_folder()

    def _browse_path(self):
        # Get last path from settings to open dialog at that location
        last_path = self.settings.value("last_browse_path", "")
        if not last_path or not os.path.exists(last_path):
            last_path = os.path.expanduser("~")
        
        path = QFileDialog.getExistingDirectory(self, "Select Folder to Scan", last_path)
        if path:
            path = os.path.normpath(path)
            # Save the selected path for next time (but don't auto-add to list)
            self.settings.setValue("last_browse_path", path)
            # Add to list if not already there
            self._add_folder_to_list(path)
    
    def _add_folder_to_list(self, path: str):
        """Add a folder to the path list if not already present."""
        items = [self.path_list_widget.item(i).data(Qt.UserRole) for i in range(self.path_list_widget.count())]
        if path not in items:
            item = QListWidgetItem("") # Keep text empty to avoid duplication with custom widget
            item.setData(Qt.UserRole, path)
            item.setSizeHint(QSize(-1, 24))
            self.path_list_widget.addItem(item)

            row_widget = QFrame()
            row_widget.setStyleSheet("background-color: transparent; border: none;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(8, 0, 4, 0)
            row_layout.setSpacing(6)

            label = QLabel(path)
            label.setStyleSheet(f"color: {MD3_COLORS['on_surface']}; font-size: 12px; background-color: transparent; border: none;")
            label.setToolTip(path)
            row_layout.addWidget(label, stretch=1)

            remove_btn = QPushButton("×")
            remove_btn.setFixedSize(18, 18)
            remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            remove_btn.setToolTip("Remove folder")
            remove_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {MD3_COLORS['on_surface_variant']};
                    border: none;
                    border-radius: 9px;
                    font-size: 14px;
                    font-weight: 600;
                    padding: 0px;
                }}
                QPushButton:hover {{
                    background-color: {MD3_COLORS['error_container']};
                    color: {MD3_COLORS['error']};
                }}
            """)
            remove_btn.clicked.connect(lambda _checked=False, item=item: self._remove_folder_item(item))
            row_layout.addWidget(remove_btn)

            self.path_list_widget.setItemWidget(item, row_widget)
    
    def _on_folders_dropped(self, paths: list):
        """Handle folders dropped on results area - supports multiple folders."""
        for path in paths:
            self._add_folder_to_list(path)
    
    def _ensure_previous_scan_stopped(self) -> bool:
        """Make sure no previous scan thread is still running before starting a new one.

        A single ScanEngine instance is reused across scans; if an old scan thread is
        still alive when a new one starts, both would mutate the engine's shared state
        (scan_cancelled, file_groups_raw, caches) concurrently. Returns True once it's
        safe to proceed, False if the old thread refused to stop in time.
        """
        old_thread = getattr(self, '_scan_thread', None)
        if old_thread is None or not old_thread.is_alive():
            return True
        self.engine.scan_cancelled = True
        old_thread.join(timeout=10.0)
        if old_thread.is_alive():
            logger.debug("DEBUG: Previous scan thread did not stop within timeout")
            dialog = CustomDialog(
                self, title="Busy",
                message="The previous scan is still shutting down. Please try again in a moment.",
                buttons=["OK"]
            )
            dialog.exec()
            return False
        return True

    def _ensure_embedding_verification_ready(self) -> bool:
        """Make sure AI Embedding verification can actually run before starting a scan
        with it enabled: onnxruntime must be installed, and the small pretrained model
        must be downloaded (with the user's consent, since it needs one internet fetch).
        Returns True once ready, False if the user should not proceed.
        """
        from core.engine import (
            ONNXRUNTIME_AVAILABLE, is_embedding_model_downloaded, download_embedding_model,
            EMBEDDING_MODEL_URL, EMBEDDING_MODEL_APPROX_SIZE_MB, EMBEDDING_MODEL_FILENAME
        )

        if not ONNXRUNTIME_AVAILABLE:
            dialog = CustomDialog(
                self, title="Missing Dependency",
                message=(
                    "AI Embedding verification requires the <b>onnxruntime</b> package, "
                    "which isn't installed.<br><br>Install it with:<br><code>pip install onnxruntime</code>"
                ),
                buttons=["OK"]
            )
            dialog.exec()
            return False

        if is_embedding_model_downloaded():
            return True

        confirm = CustomDialog(
            self, title="Download AI Model",
            message=(
                f"AI Embedding verification needs a one-time download of a small pretrained "
                f"vision model (<b>{EMBEDDING_MODEL_FILENAME}</b>, ~{EMBEDDING_MODEL_APPROX_SIZE_MB}MB) from:"
                f"<br>{EMBEDDING_MODEL_URL}<br><br>"
                f"It will be cached locally afterward &mdash; no further downloads needed. Download now?"
            ),
            buttons=["Cancel", "Download"]
        )
        if confirm.exec() != "Download":
            return False

        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("CloneWiper")
        progress_dialog.setModal(True)
        progress_dialog.setFixedSize(380, 120)
        layout = QVBoxLayout(progress_dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        label = QLabel(f"Downloading AI model (~{EMBEDDING_MODEL_APPROX_SIZE_MB}MB)...")
        bar = QProgressBar()
        bar.setRange(0, 100)
        layout.addWidget(label)
        layout.addWidget(bar)

        class _DownloadSignals(QObject):
            progress = Signal(float)
            done = Signal(bool)

        signals = _DownloadSignals()
        signals.progress.connect(lambda p: bar.setValue(int(p * 100)))

        outcome = {'success': False}

        def _on_done(success: bool):
            outcome['success'] = success
            progress_dialog.accept()

        signals.done.connect(_on_done)

        def _worker():
            ok = download_embedding_model(progress_callback=signals.progress.emit)
            signals.done.emit(ok)

        threading.Thread(target=_worker, daemon=True).start()
        progress_dialog.exec()

        if not outcome['success']:
            dialog = CustomDialog(
                self, title="Download Failed",
                message="Could not download the AI model. Check your internet connection and try again.",
                buttons=["OK"]
            )
            dialog.exec()
            return False
        return True

    def _start_scanning(self):
        if not self._ensure_previous_scan_stopped():
            return
        paths = [self.path_list_widget.item(i).data(Qt.UserRole) for i in range(self.path_list_widget.count())]
        if not paths:
            dialog = CustomDialog(self, title="Warning", message="Please select at least one folder to scan.", buttons=["OK"])
            dialog.exec()
            return
        
        valid_paths = [p for p in paths if os.path.exists(p)]
        
        if not valid_paths:
            dialog = CustomDialog(self, title="Error", message="No valid paths found!", buttons=["OK"])
            dialog.exec()
            return
        
        self.scan_paths = valid_paths

        # Get hash mode: 0=MD5, 1=Single Perceptual, 2=Multi-Algorithm, 3=Single+ORB,
        # 4=Multi+ORB, 5=Multi+AI Embedding
        hash_mode = self.hash_mode_combo.currentIndex()
        use_imagehash = 0 < hash_mode < 6  # True for perceptual hash modes
        use_multi_hash = hash_mode in (2, 4, 5)  # True for multi-algorithm modes
        use_orb_verification = hash_mode == 3 or hash_mode == 4  # True for ORB modes
        use_embedding_verification = hash_mode == 5  # True for AI embedding mode

        if use_embedding_verification and not self._ensure_embedding_verification_ready():
            return

        # Update button to "Cancel"
        self.scan_btn.setText("Cancel")
        self.scan_btn.setEnabled(True)
        # Disconnect previous connection and connect to cancel function
        self.scan_btn.clicked.disconnect()
        self.scan_btn.clicked.connect(self._cancel_scanning)

        # Center progress container will be shown via status callback
        # Don't update status_label during scanning to avoid duplication with center display

        # Clear previous results
        self.file_groups = {}
        self.file_groups_raw = {}
        # Clear selection state when starting new scan
        self.selection_state.clear()
        # Reset delete button counter
        self._update_delete_ui()
        # Reset page navigation
        self.current_page = 0
        self.page_combo.blockSignals(True)
        self.page_combo.set_total_pages(1)  # Set total pages for display
        self.page_combo.clear()
        self.page_combo.addItem("1")
        self.page_combo.setCurrentIndex(0)
        self.page_combo.blockSignals(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        # Hide image-specific buttons when starting new scan
        if hasattr(self, 'keep_best_btn'):
            self.keep_best_btn.setVisible(False)
        if hasattr(self, 'keep_raw_btn'):
            self.keep_raw_btn.setVisible(False)
        
        # Reset quick-select highlight for new scan
        self._highlight_quick_select_mode(None)
        
        # Clear existing group widgets (but keep center_progress_container and stretches)
        # Reset status label to initial instruction when clearing results
        self.status_label.setText("")
        self.status_bar.setVisible(False)
        items_to_remove = []
        for i in range(self.results_layout.count()):
            item = self.results_layout.itemAt(i)
            if item and item.widget() and item.widget() != self.center_progress_container:
                items_to_remove.append(i)
        # Remove in reverse order to maintain indices
        for i in reversed(items_to_remove):
            item = self.results_layout.takeAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        
        # Show center progress container with progress bar and hide upper left status label
        self.center_progress_container.setVisible(True)
        self.center_status_label.setText("Initializing scan...")
        self.center_progress_bar.setVisible(True)  # Show progress bar during scanning
        self.center_progress_bar.setValue(0)
        self._scan_status_message = "Initializing scan..."
        self.status_label.setVisible(False)
        self.status_bar.setVisible(False)
        
        # Hide footer during scanning
        if hasattr(self, 'footer'):
            self.footer.setVisible(False)
        
        # Store scan thread reference for potential cancellation
        self._scan_thread = None
        
        # Run scan in background thread
        def scan_thread():
            try:
                logger.debug(f"DEBUG: Starting scan thread for paths: {valid_paths}")
                logger.debug(f"DEBUG: use_imagehash={use_imagehash}")
                # Ensure callbacks are set
                logger.debug(f"DEBUG: status_callback={self.engine.status_callback}")
                logger.debug(f"DEBUG: progress_callback={self.engine.progress_callback}")
                logger.debug(f"DEBUG: results_callback={self.engine.results_callback}")
                
                # Test callback immediately
                self.engine.status_callback("Scan thread started...")
                self.engine.progress_callback(0.01)
                
                self.engine.scan_duplicate_files(valid_paths, use_imagehash, use_multi_hash, use_orb_verification, use_embedding_verification)
                logger.debug("DEBUG: Scan completed")
            except Exception as e:
                logger.debug("Scan thread exception: %s", e, exc_info=True)
                # Report error via status callback
                self.engine.status_callback(f"Scan error: {e}")
                self.engine.results_callback({})
        
        self._scan_thread = threading.Thread(target=scan_thread, daemon=True)
        self._scan_thread.start()
        logger.debug(f"DEBUG: Scan thread started, thread={self._scan_thread}")
    
    def _on_progress(self, value: float):
        """Progress callback from engine (thread-safe via Signal)."""
        # Emit signal for thread-safe UI update
        self.progress_updated.emit(value)
    
    def _on_status(self, message: str):
        """Status callback from engine (thread-safe via Signal)."""
        # Emit signal for thread-safe UI update
        self.status_updated.emit(str(message))
    
    def _on_results(self, duplicate_groups: Dict[str, List[str]]):
        """Results callback from engine (thread-safe via Signal)."""
        # Emit signal for thread-safe UI update
        self.results_ready.emit(duplicate_groups)
    
    @Slot(float)
    def _on_progress_slot(self, value: float):
        """Slot for progress updates (runs on main thread)."""
        if self.scan_btn.text() == "Cancel":
            self.center_progress_bar.setValue(int(value * 100))
            self._refresh_center_scan_display()

    def _refresh_center_scan_display(self) -> None:
        """Keep status text and progress bar in sync during scans."""
        if self.scan_btn.text() != "Cancel":
            return
        pct = self.center_progress_bar.value()
        message = getattr(self, "_scan_status_message", "")
        if pct > 0 and message:
            self.center_status_label.setText(f"{message} ({pct}%)")
        elif message:
            self.center_status_label.setText(message)
    
    @Slot(str)
    def _on_status_slot(self, message: str):
        """Slot for status updates (runs on main thread)."""
        if self.scan_btn.text() == "Cancel":
            self._scan_status_message = message
            self.center_progress_container.setVisible(True)
            self.center_progress_bar.setVisible(True)
            self.status_label.setVisible(False)
            self.status_bar.setVisible(False)
            self._refresh_center_scan_display()
        else:
            self.status_label.setText(message)
            self.status_label.setVisible(bool(message))
            self.status_bar.setVisible(bool(message))
            self.center_progress_container.setVisible(False)
            self.center_progress_bar.setVisible(False)
    
    @Slot(dict)
    def _on_results_slot(self, duplicate_groups: Dict[str, List[str]]):
        """Slot for results updates (runs on main thread)."""
        # Cache maintenance doesn't need to happen before the user sees their results;
        # run it in the background instead of blocking the GUI thread right when
        # results are ready to render.
        threading.Thread(target=self._cleanup_thumbnail_cache, daemon=True).start()
        self._display_results(duplicate_groups)

    def _cleanup_thumbnail_cache(self, vacuum: bool = False):
        """Keep persistent cache focused on recent expensive media thumbnails."""
        try:
            self.persistent_cache.restrict_to_extensions(ThumbnailWorker.PERSISTENT_CACHE_EXTS)
            self.persistent_cache.cleanup(max_days=30, vacuum=vacuum)
        except Exception as e:
            logger.debug("Thumbnail cache cleanup failed: %s", e)
    
    
    def _cancel_scanning(self):
        """Cancel the current scanning process."""
        logger.debug("DEBUG: Cancel scanning requested")
        # Set cancellation flag in engine
        self.engine.scan_cancelled = True
        
        # Show instruction again after cancellation
        self.center_progress_container.setVisible(True)
        self.center_status_label.setText(self._get_instruction_text())
        self.center_progress_bar.setVisible(False)  # Hide progress bar
        
        # Update UI
        self.scan_btn.setText("Scan")
        self.scan_btn.setEnabled(True)
        # Reconnect button to start scanning
        self.scan_btn.clicked.disconnect()
        self.scan_btn.clicked.connect(self._start_scanning)
        
        self.status_label.setVisible(False)
        self.status_bar.setVisible(False)
        
        # Hide footer on cancel
        if hasattr(self, 'footer'):
            self.footer.setVisible(False)

        # Note: intentionally not clearing self._scan_thread here. The thread is only
        # cooperatively cancelled (checks scan_cancelled periodically) and may still be
        # running; _ensure_previous_scan_stopped() needs this reference to join it before
        # a new scan starts.

    def _display_results(self, duplicate_groups: Dict[str, List[str]]):
        """Display scan results."""
        logger.debug(f"DEBUG: _display_results called with {len(duplicate_groups)} groups")
        
        # Hide center progress container and show results
        self.center_progress_container.setVisible(False)
        self.center_progress_bar.setVisible(False)  # Hide progress bar
        
        # Restore button to "Start Scanning" state
        self.scan_btn.setText("Scan")
        self.scan_btn.setEnabled(True)
        # Reconnect button to start scanning
        self.scan_btn.clicked.disconnect()
        self.scan_btn.clicked.connect(self._start_scanning)
        
        # Copy rather than alias: this dict is also held by the engine as its own
        # file_groups_raw, so keeping the same object risks the UI iterating it while
        # a stray scan thread mutates it.
        self.file_groups_raw = dict(duplicate_groups)

        # Apply current sorting
        current_sort = self.sort_combo.currentText()
        logger.debug(f"DEBUG: Applying current sorting: {current_sort}")
        self.file_groups = self.engine.apply_sorting(
            duplicate_groups,
            current_sort,
            group_by_type=False
        )
        
        logger.debug(f"DEBUG: Enabling scan button, hiding progress bar")
        self.center_progress_bar.setValue(0)
        
        if not duplicate_groups:
            self._results_summary = ""
            self.center_progress_container.setVisible(True)
            self.center_progress_bar.setVisible(False)
            self.center_status_label.setText(
                "<div style='text-align: center;'>"
                "<p style='font-size: 24px; font-weight: 600; margin: 0 0 8px 0;'>"
                "No duplicates in these folders"
                "</p>"
                "<p style='font-size: 14px; color: #CAC4D0; margin: 0;'>"
                "Try another folder, or switch to a broader photo mode if you expected similar images."
                "</p>"
                "</div>"
            )
            self.status_label.setVisible(False)
            self.status_bar.setVisible(False)
            if hasattr(self, 'footer'):
                self.footer.setVisible(False)
            self.keep_best_btn.setVisible(False)
            self.keep_smallest_btn.setVisible(False)
            self.keep_raw_btn.setVisible(False)
            return

        self._results_summary = self._build_results_summary(duplicate_groups)
        self.status_label.setText(self._results_summary)
        self.status_label.setVisible(True)
        self.status_bar.setVisible(True)
        if hasattr(self, 'footer'):
            self.footer.setVisible(True)
        
        self._update_image_button_visibility()
        
        try:
            self._render_page(0)
        except Exception as e:
            logger.debug("Error in _render_page: %s", e, exc_info=True)
        logger.debug(f"DEBUG: _display_results completed")

    def _build_results_summary(self, groups: Dict[str, List[str]]) -> str:
        """One-line result summary: counts plus space recoverable by keeping one file."""
        file_count = 0
        reclaim = 0
        for files in groups.values():
            sizes = []
            for file_path in files:
                try:
                    sizes.append(os.path.getsize(file_path))
                except OSError:
                    continue
            file_count += len(files)
            if len(sizes) > 1:
                reclaim += sum(sizes) - max(sizes)
        group_word = "group" if len(groups) == 1 else "groups"
        file_word = "file" if file_count == 1 else "files"
        summary = f"{len(groups)} {group_word}  ·  {file_count} {file_word}"
        if reclaim > 0:
            summary += f"  ·  {format_byte_size(reclaim)} reclaimable if you keep one file per group"
        return summary
    
    def _render_page(self, page_index: int):
        """Render a page of groups."""
        # Reset quick-select highlight if changing page and not in "All Pages" mode
        if hasattr(self, 'scope_toggle') and not self.scope_toggle.isChecked():
            if page_index != self.current_page:
                self._highlight_quick_select_mode(None)

        logger.debug(f"DEBUG: _render_page called with page_index={page_index}, file_groups={len(self.file_groups) if self.file_groups else 0}")
        # Clear existing group widgets (but keep center_progress_container and stretches)
        self._group_widgets.clear()
        # Drop references to the previous page's thumbnail-worker signal objects. They were
        # only kept alive to prevent GC while their (now superseded) GroupWidgets were on
        # screen; retaining them forever leaked a Signals object + closure per thumbnail ever
        # requested in the app's lifetime.
        if hasattr(self, '_active_signals'):
            self._active_signals.clear()
        logger.debug(f"DEBUG: Clearing existing group widgets")
        # Remove all widgets except center_progress_container and stretches
        items_to_remove = []
        for i in range(self.results_layout.count()):
            item = self.results_layout.itemAt(i)
            if item and item.widget() and item.widget() != self.center_progress_container:
                items_to_remove.append(i)
        # Remove in reverse order to maintain indices
        for i in reversed(items_to_remove):
            item = self.results_layout.takeAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        
        if not self.file_groups:
            logger.debug(f"DEBUG: No file_groups to render")
            return
        
        all_groups = list(self.file_groups.items())
        total = len(all_groups)
        logger.debug(f"DEBUG: Total groups: {total}")
        max_page = (total - 1) // self.groups_per_page if total else 0
        page_index = max(0, min(page_index, max_page))
        self.current_page = page_index
        
        start = page_index * self.groups_per_page
        end = min(total, start + self.groups_per_page)
        page_groups = all_groups[start:end]
        logger.debug(f"DEBUG: Rendering groups {start} to {end} ({len(page_groups)} groups)")
        
        # Update pager
        total_pages = max_page + 1 if total > 0 else 1
        
        # Update page combo box
        # Dropdown items show only page numbers, but button displays "current/total"
        # Block signals to prevent triggering navigation when updating programmatically
        self.page_combo.blockSignals(True)
        self.page_combo.set_total_pages(total_pages)  # Set total pages for display
        self.page_combo.clear()
        if total_pages > 0:
            for i in range(1, total_pages + 1):
                self.page_combo.addItem(str(i))  # Dropdown items show only page numbers
        else:
            self.page_combo.addItem("1")
        self.page_combo.setCurrentIndex(page_index)
        self.page_combo.blockSignals(False)
        
        self.prev_btn.setEnabled(page_index > 0)
        self.next_btn.setEnabled(page_index < max_page)
        
        # Create group widgets
        self._thumb_page_token = time.monotonic()
        
        # Reset thumbnail loading tracking
        self._pending_thumbnails.clear()
        self._thumbnail_total = 0
        self._thumbnail_loaded = 0
        
        # Count total thumbnails to load (PDF/EPUB/MOBI/AZW3 files)
        for idx, (group_id, files) in enumerate(page_groups):
            for file_path in files:
                if file_path not in self.thumb_cache:
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in {'.pdf', '.epub', '.mobi', '.azw3'}:
                        self._thumbnail_total += 1
                        self._pending_thumbnails[file_path] = False
        
        for idx, (group_id, files) in enumerate(page_groups):
            try:
                group_widget = GroupWidget(group_id, files, group_index=start + idx, total_groups=total)
                group_widget.selection_changed.connect(self._on_selection_changed)
                self.results_layout.insertWidget(self.results_layout.count() - 1, group_widget)
                self._group_widgets[group_id] = group_widget
                
                # Restore selection state for files in this group
                for file_path in files:
                    if file_path in self.selection_state:
                        group_widget.set_selection(file_path, self.selection_state[file_path])
                
                # Request thumbnails
                for file_path in files:
                    self._request_thumbnail(file_path, group_widget)
            except Exception as e:
                logger.debug("Error creating GroupWidget for %s: %s", group_id, e, exc_info=True)
        
        # Reset scroll to top
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(0))
        
        # Set initial status - show thumbnail loading if applicable, otherwise show groups info
        summary = getattr(self, "_results_summary", "")
        page_label = f"Page {page_index + 1} of {total_pages}"
        if self._thumbnail_total > 0:
            status_text = f"{summary}   ·   Loading previews 0/{self._thumbnail_total}" if summary else f"Loading previews 0/{self._thumbnail_total}"
        else:
            status_text = f"{summary}   ·   {page_label}" if summary else page_label
        self.status_label.setText(status_text.strip())
        self.status_bar.setVisible(True)
        logger.debug(f"DEBUG: _render_page completed, created {len(self._group_widgets)} group widgets")
        
        # Force update
        self.results_container.update()
        self.scroll_area.update()
    
    def _request_thumbnail(self, file_path: str, group_widget: GroupWidget):
        """Request thumbnail load for a file."""
        # Check cache
        if file_path in self.thumb_cache:
            pixmap = self.thumb_cache[file_path]
            aspect = self.thumb_aspects.get(file_path, 1.0)
            group_widget.set_thumbnail(file_path, pixmap, aspect)
            if file_path in self.metadata_cache:
                group_widget.set_metadata(file_path, self.metadata_cache[file_path])
            return
        
        # Check if this is a PDF/EPUB/MOBI/AZW3 file that needs loading status
        ext = os.path.splitext(file_path)[1].lower()
        is_document = ext in {'.pdf', '.epub', '.mobi', '.azw3'}
        
        if is_document and file_path in self._pending_thumbnails:
            self._pending_thumbnails[file_path] = True  # Mark as loading
            # Update status if we have document thumbnails to load
            if self._thumbnail_total > 0:
                # Count how many have started loading (marked as True)
                started_count = sum(1 for loading in self._pending_thumbnails.values() if loading)
                # Show progress: started loading / total
                detail = f"Loading previews {started_count}/{self._thumbnail_total}"
                summary = getattr(self, "_results_summary", "")
                self.status_label.setText(f"{summary}   ·   {detail}" if summary else detail)
        
        # Submit worker
        worker = ThumbnailWorker(file_path, 300, self._thumb_page_token, cache=self.persistent_cache)
        
        # Store signals to prevent GC
        if not hasattr(self, '_active_signals'):
            self._active_signals = set()
        
        # Connect signals (using QueuedConnection implicitly via lambda onto main thread)
        worker.signals.thumb_ready.connect(lambda p, pix, asp: self._on_thumb_ready(p, pix, asp, group_widget))
        worker.signals.meta_ready.connect(lambda p, m: self._on_meta_ready(p, m, group_widget))
        
        # Keep the signals in a set for the duration of the page
        self._active_signals.add(worker.signals)
        
        self.thread_pool.start(worker)
    
    def _on_thumb_ready(self, path: str, pixmap: QPixmap, aspect: float, group_widget: Optional[GroupWidget] = None):
        """Handle thumbnail ready signal."""
        logger.debug(f"DEBUG: _on_thumb_ready called for {os.path.basename(path)}, aspect={aspect:.2f}, pixmap size={pixmap.size().width()}x{pixmap.size().height()}, group_widget={group_widget is not None}")
        
        # Update thumbnail loading progress for PDF/EPUB/MOBI/AZW3 files
        ext = os.path.splitext(path)[1].lower()
        is_document = ext in {'.pdf', '.epub', '.mobi', '.azw3'}
        
        if is_document and path in self._pending_thumbnails:
            self._thumbnail_loaded += 1
            del self._pending_thumbnails[path]
            
            # Update status
            if self._thumbnail_total > 0:
                remaining = self._thumbnail_total - self._thumbnail_loaded
                if remaining > 0:
                    detail = f"Loading previews {self._thumbnail_loaded}/{self._thumbnail_total}"
                    summary = getattr(self, "_results_summary", "")
                    self.status_label.setText(f"{summary}   ·   {detail}" if summary else detail)
                else:
                    if self.file_groups:
                        total_pages = max(0, (len(self.file_groups) - 1) // self.groups_per_page) + 1
                        summary = getattr(self, "_results_summary", "")
                        page_label = f"Page {self.current_page + 1} of {total_pages}"
                        self.status_label.setText(f"{summary}   ·   {page_label}" if summary else page_label)
        
        # Check if stale - if group_widget is provided and still valid, always accept
        # Otherwise, check time difference (allow up to 60 seconds for slow loading)
        if group_widget is None:
            current_time = time.monotonic()
            time_diff = abs(current_time - self._thumb_page_token)
            if time_diff > 60.0:
                logger.debug(f"DEBUG: Thumbnail for {os.path.basename(path)} is stale (diff={time_diff:.2f}s), skipping")
                return
        
        # Cache
        self.thumb_cache[path] = pixmap
        self.thumb_aspects[path] = aspect
        if len(self.thumb_cache) > self.thumb_cache_max:
            oldest = next(iter(self.thumb_cache))
            del self.thumb_cache[oldest]
            self.thumb_aspects.pop(oldest, None)
        
        # Update widget (find by path if not provided)
        if group_widget:
            # Verify widget is still in active widgets before updating
            if group_widget in self._group_widgets.values():
                try:
                    group_widget.set_thumbnail(path, pixmap, aspect)
                    logger.debug(f"DEBUG: Updated thumbnail for {os.path.basename(path)} in provided group widget")
                except RuntimeError:
                    # Widget was deleted, ignore
                    pass
        else:
            # Find widget containing this file
            for widget in list(self._group_widgets.values()):  # Create a copy to avoid modification during iteration
                if path in widget.files:
                    try:
                        widget.set_thumbnail(path, pixmap, aspect)
                        logger.debug(f"DEBUG: Updated thumbnail for {os.path.basename(path)} in found group widget")
                        break
                    except RuntimeError:
                        # Widget was deleted, continue to next
                        continue
    
    def _on_meta_ready(self, path: str, meta: str, group_widget: Optional[GroupWidget] = None):
        """Handle metadata ready signal."""
        logger.debug(f"DEBUG: _on_meta_ready called for {os.path.basename(path)}, meta={meta}, group_widget={group_widget is not None}")
        self.metadata_cache[path] = meta
        if len(self.metadata_cache) > self.metadata_cache_max:
            oldest = next(iter(self.metadata_cache))
            del self.metadata_cache[oldest]
        
        if group_widget:
            # Verify widget is still in active widgets before updating
            if group_widget in self._group_widgets.values():
                try:
                    group_widget.set_metadata(path, meta)
                except RuntimeError:
                    # Widget was deleted, ignore
                    pass
        else:
            # Find widget containing this file
            for widget in list(self._group_widgets.values()):  # Create a copy to avoid modification during iteration
                if path in widget.files:
                    try:
                        widget.set_metadata(path, meta)
                        break
                    except RuntimeError:
                        # Widget was deleted, continue to next
                        continue
    
    def _prev_page(self):
        if self.current_page > 0:
            self._render_page(self.current_page - 1)
    
    def _next_page(self):
        if self.file_groups:
            total = len(self.file_groups)
            max_page = (total - 1) // self.groups_per_page
            if self.current_page < max_page:
                self._render_page(self.current_page + 1)
    
    def _on_page_selected(self, index: int):
        """Handle page selection from combo box."""
        if index >= 0 and self.file_groups:
            self._render_page(index)
    
    def _refresh_display(self):
        """Refresh current page, preserving scroll position."""
        # Save scroll anchor
        if hasattr(self, 'scroll_area') and self.scroll_area:
            scroll_bar = self.scroll_area.verticalScrollBar()
            scroll_value = scroll_bar.value()
            # Find first visible group
            visible_group_id = None
            for group_id, widget in self._group_widgets.items():
                if widget.isVisible():
                    visible_group_id = group_id
                    break
            if visible_group_id:
                self._scroll_anchor = (visible_group_id, scroll_value)
        
        # Re-render
        self._render_page(self.current_page)
        
        # Restore scroll after layout
        QTimer.singleShot(100, self._restore_scroll_anchor)
    
    def _restore_scroll_anchor(self):
        """Restore scroll position after refresh."""
        if not self._scroll_anchor:
            return
        
        group_id, target_value = self._scroll_anchor
        self._scroll_anchor = None
        
        if hasattr(self, 'scroll_area') and self.scroll_area and group_id in self._group_widgets:
            widget = self._group_widgets[group_id]
            self.scroll_area.ensureWidgetVisible(widget)
            scroll_bar = self.scroll_area.verticalScrollBar()
            scroll_bar.setValue(target_value)
    
    def _apply_sorting(self, sort_value: str):
        """Apply sorting to results."""
        if not self.file_groups_raw:
            return
        
        sorted_groups = self.engine.apply_sorting(
            self.file_groups_raw,
            sort_value,
            group_by_type=False  # Can add toggle later
        )
        self.file_groups = sorted_groups
        self._render_page(0)
    
    @Slot(str, bool)
    def _on_selection_changed(self, path: str, selected: bool):
        """Handle manual selection change from a group widget."""
        self.selection_state[path] = selected
        self._update_delete_ui()

    def _clear_selection(self):
        """Unselect all files marked for deletion."""
        self.selection_state.clear()
        
        # Update all visible widgets
        for widget in self._group_widgets.values():
            for i in range(widget.model.rowCount()):
                path = widget.model.index(i).data(Qt.ItemDataRole.UserRole)
                widget.set_selection(path, False)
        
        self._update_delete_ui()

    def mousePressEvent(self, event):
        if self._is_windows:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            edge = self._get_resize_edge(event.pos())
            if edge:
                self._resize_edge_active = edge
                self._resize_start_pos = event.globalPos()
                self._resize_start_geom = self.geometry()
                event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_windows:
            super().mouseMoveEvent(event)
            return
        edge = self._get_resize_edge(event.pos())
        if edge:
            if edge in ('left', 'right'): self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif edge in ('top', 'bottom'): self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif edge in ('top_left', 'bottom_right'): self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            else: self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.unsetCursor()

        if self._resize_edge_active:
            delta = event.globalPos() - self._resize_start_pos
            new_geom = QRect(self._resize_start_geom)
            
            if 'left' in self._resize_edge_active:
                new_geom.setLeft(self._resize_start_geom.left() + delta.x())
            if 'right' in self._resize_edge_active:
                new_geom.setRight(self._resize_start_geom.right() + delta.x())
            if 'top' in self._resize_edge_active:
                new_geom.setTop(self._resize_start_geom.top() + delta.y())
            if 'bottom' in self._resize_edge_active:
                new_geom.setBottom(self._resize_start_geom.bottom() + delta.y())
            
            if new_geom.width() >= self.minimumWidth() and new_geom.height() >= self.minimumHeight():
                self.setGeometry(new_geom)
            event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._resize_edge_active = False
        super().mouseReleaseEvent(event)

    def _get_resize_edge(self, pos):
        margin = self._resize_border
        w, h = self.width(), self.height()
        x, y = pos.x(), pos.y()
        
        edge = []
        if x < margin: edge.append('left')
        elif x > w - margin: edge.append('right')
        if y < margin: edge.append('top')
        elif y > h - margin: edge.append('bottom')
        
        return '_'.join(edge) if edge else None

    def nativeEvent(self, event_type, message):
        """Use Windows hit testing so frameless chrome still supports native resize/snap."""
        if not self._is_windows:
            return super().nativeEvent(event_type, message)

        msg = ctypes.wintypes.MSG.from_address(int(message))
        wm_nchittest = 0x0084
        wm_nclbuttondown = 0x00A1
        wm_nclbuttonup = 0x00A2
        if msg.message in (wm_nclbuttondown, wm_nclbuttonup) and msg.wParam == self._win_ht_values["HTMAXBUTTON"]:
            if msg.message == wm_nclbuttonup and hasattr(self, "custom_title_bar"):
                x = ctypes.c_short(msg.lParam & 0xFFFF).value
                y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
                max_button_pos = self.custom_title_bar.max_btn.mapFromGlobal(QPoint(x, y))
                if self.custom_title_bar.max_btn.rect().contains(max_button_pos):
                    self.custom_title_bar._toggle_maximize()
            return True, 0

        if msg.message != wm_nchittest:
            return super().nativeEvent(event_type, message)

        x = ctypes.c_short(msg.lParam & 0xFFFF).value
        y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
        pos = self.mapFromGlobal(QPoint(x, y))
        hit_test = self._native_hit_test(pos, QPoint(x, y))
        if hit_test is None:
            return super().nativeEvent(event_type, message)
        return True, hit_test

    def _native_hit_test(self, pos: QPoint, global_pos: QPoint):
        """Return Windows HT* codes for resize borders, caption drag, and snap layout."""
        if not self.rect().contains(pos):
            return None

        ht = self._win_ht_values
        if not (self.isMaximized() or self.isFullScreen()):
            margin = self._resize_border
            left = pos.x() < margin
            right = pos.x() >= self.width() - margin
            top = pos.y() < margin
            bottom = pos.y() >= self.height() - margin

            if top and left:
                return ht["HTTOPLEFT"]
            if top and right:
                return ht["HTTOPRIGHT"]
            if bottom and left:
                return ht["HTBOTTOMLEFT"]
            if bottom and right:
                return ht["HTBOTTOMRIGHT"]
            if left:
                return ht["HTLEFT"]
            if right:
                return ht["HTRIGHT"]
            if top:
                return ht["HTTOP"]
            if bottom:
                return ht["HTBOTTOM"]

        if hasattr(self, "custom_title_bar") and self.custom_title_bar.geometry().contains(pos):
            max_button_pos = self.custom_title_bar.max_btn.mapFromGlobal(global_pos)
            if self.custom_title_bar.max_btn.rect().contains(max_button_pos):
                return ht["HTMAXBUTTON"]

            title_bar_pos = self.custom_title_bar.mapFromGlobal(global_pos)
            child = self.custom_title_bar.childAt(title_bar_pos)
            if child in (self.custom_title_bar.min_btn, self.custom_title_bar.max_btn, self.custom_title_bar.close_btn):
                return ht["HTCLIENT"]
            return ht["HTCAPTION"]

        return None
    
    def changeEvent(self, event):
        """Handle window state changes (maximized/restored) to toggle rounded corners."""
        if event.type() == event.Type.WindowStateChange:
            is_maximized = self.isMaximized() or self.isFullScreen()
            if hasattr(self, "custom_title_bar"):
                self.custom_title_bar.set_maximized_state(is_maximized)
            if is_maximized:
                # Remove rounded corners when maximized/fullscreen
                self.setMask(QRegion())
            else:
                # Apply rounded corners when not maximized
                self._apply_rounded_corners()
        super().changeEvent(event)
    
    def resizeEvent(self, event):
        """Update rounded corners mask on resize."""
        super().resizeEvent(event)
        if not (self.isMaximized() or self.isFullScreen()):
            self._apply_rounded_corners()
    
    def showEvent(self, event):
        """Apply rounded corners when window is first shown."""
        super().showEvent(event)
        if not (self.isMaximized() or self.isFullScreen()):
            self._apply_rounded_corners()
    
    def _apply_rounded_corners(self):
        """Apply rounded corners mask to the window."""
        if self._is_windows:
            try:
                # DWMWA_WINDOW_CORNER_PREFERENCE = 33
                # DWMWCP_ROUND = 2
                hwnd = self.winId()
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, 33, ctypes.byref(ctypes.c_int(2)), 4
                )
            except Exception:
                pass

        path = QPainterPath()
        rect = self.rect()
        path.addRoundedRect(rect, self._corner_radius, self._corner_radius)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)

    def _update_delete_ui(self):
        """Update the delete and clear selection buttons based on state."""
        selected_count = sum(1 for v in self.selection_state.values() if v)
        self.delete_btn.setEnabled(selected_count > 0)
        if selected_count > 0:
            self.delete_btn.setText(f"Delete {selected_count}")
        else:
            self.delete_btn.setText("Delete selected")
        
        # Also update clear selection button
        if hasattr(self, 'clear_select_btn'):
            self.clear_select_btn.setEnabled(selected_count > 0)
    
    def _update_image_button_visibility(self):
        """Update visibility of Keep Best, Keep Smallest, and Keep RAW buttons based on applicable files."""
        if not self.file_groups:
            self.keep_best_btn.setVisible(False)
            self.keep_smallest_btn.setVisible(False)
            self.keep_raw_btn.setVisible(False)
            return
        
        # Check if there are any groups with at least 2 image files (for Keep Best)
        has_applicable_images = False
        for hash_val, files in self.file_groups.items():
            if len(files) < 2:
                continue
            img_files = [fp for fp in files if self.engine.is_image_file(fp)]
            if len(img_files) >= 2:
                has_applicable_images = True
                break
        
        # Check if there are any groups with RAW files mixed with non-RAW images (for Keep RAW)
        has_raw_mixed = False
        for hash_val, files in self.file_groups.items():
            if len(files) < 2:
                continue
            img_files = [fp for fp in files if self.engine.is_image_file(fp)]
            if len(img_files) < 2:
                continue
            raw_files = [fp for fp in img_files if self.engine.is_raw_image_file(fp)]
            non_raw_files = [fp for fp in img_files if not self.engine.is_raw_image_file(fp)]
            # Only show Keep RAW if there are both RAW and non-RAW files in the same group
            if raw_files and non_raw_files:
                has_raw_mixed = True
                break
        
        self.keep_best_btn.setVisible(has_applicable_images)
        self.keep_smallest_btn.setVisible(has_applicable_images)
        self.keep_raw_btn.setVisible(has_raw_mixed)

    def _update_scope_label(self, apply_all_pages: bool):
        """Keep quick-select scope wording aligned with the toggle state."""
        if hasattr(self, "scope_label"):
            self.scope_label.setText("All results" if apply_all_pages else "This page")

    def _highlight_quick_select_mode(self, mode: str):
        """Visually mark the last quick-select strategy the user applied."""
        if not hasattr(self, "quick_select_buttons"):
            return
        for button_mode, button in self.quick_select_buttons.items():
            button.setProperty("quickSelected", button_mode == mode)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _quick_select(self, mode: str):
        """Apply quick selection."""
        if not self.file_groups:
            return
        
        apply_all = self.scope_toggle.isChecked()
        current_page_groups = None
        if not apply_all:
            all_items = list(self.file_groups.items())
            start = self.current_page * self.groups_per_page
            end = min(len(all_items), start + self.groups_per_page)
            current_page_groups = all_items[start:end]
        
        decisions = self.engine.compute_quick_select_decisions(
            self.file_groups,
            mode,
            apply_all,
            current_page_groups
        )
        
        # Update selection state
        for path, should_delete in decisions.items():
            self.selection_state[path] = should_delete
        
        # Update visible widgets
        for group_id, widget in self._group_widgets.items():
            for path in widget.files:
                if path in decisions:
                    widget.set_selection(path, decisions[path])
        
        # Update UI
        self._highlight_quick_select_mode(mode)
        self._update_delete_ui()
        
        scope_text = "All groups" if apply_all else "Current page"
        message = f"Scope: <b>{scope_text}</b><br>Processed <b>{len(decisions)}</b> files."
        dialog = CustomDialog(self, title="Quick Select Complete", message=message, buttons=["OK"])
        dialog.exec()
    
    def _delete_selected(self):
        """Delete selected files."""
        selected = [path for path, delete in self.selection_state.items() if delete]
        if not selected:
            return
        
        dialog = CustomDialog(self, title="Confirm Delete", message=f"Are you sure you want to delete {len(selected)} files?", buttons=["Yes", "No"])
        result = dialog.exec()
        
        if result == "Yes":
            try:
                import send2trash
                deleted_paths = set(selected)
                failed_paths = []

                try:
                    # send2trash accepts a list of paths on Windows, which avoids one COM
                    # file operation per file and is much faster for large selections.
                    send2trash.send2trash(selected)
                    deleted_count = len(selected)
                except Exception as batch_error:
                    logger.warning("Batch delete failed, falling back to per-file delete: %s", batch_error)
                    deleted_paths.clear()
                    for path in selected:
                        try:
                            send2trash.send2trash(path)
                            deleted_paths.add(path)
                        except Exception as e:
                            logger.warning("Failed to delete %s: %s", path, e)
                            failed_paths.append(path)
                    deleted_count = len(deleted_paths)
                failed_count = len(failed_paths)

                if not deleted_paths and failed_count:
                    message = f"No files were deleted.\n{failed_count} files failed to delete."
                    dialog = CustomDialog(self, title="Delete Failed", message=message, buttons=["OK"])
                    dialog.exec()
                    return

                def keep_duplicate_groups(groups):
                    updated_groups = {}
                    for group_id, files in groups.items():
                        remaining_files = [f for f in files if f not in deleted_paths]
                        if len(remaining_files) > 1:
                            updated_groups[group_id] = remaining_files
                    return updated_groups

                # Remove deleted files from file_groups and file_groups_raw without
                # extra exists() calls; the selected set already tells us what changed.
                if self.file_groups:
                    self.file_groups = keep_duplicate_groups(self.file_groups)
                
                if self.file_groups_raw:
                    self.file_groups_raw = keep_duplicate_groups(self.file_groups_raw)
                
                if deleted_paths:
                    self.persistent_cache.remove_many(deleted_paths)
                    for path in deleted_paths:
                        self.thumb_cache.pop(path, None)
                        self.thumb_aspects.pop(path, None)
                        self.metadata_cache.pop(path, None)
                
                # Clear selection state for deleted files
                for path in deleted_paths:
                    self.selection_state.pop(path, None)
                for path in failed_paths:
                    try:
                        self.selection_state[path] = True
                    except Exception:
                        pass
                
                # Update delete button UI
                self._update_delete_ui()
                
                # Show result message
                if failed_count > 0:
                    message = f"Deleted {deleted_count} files.\n{failed_count} files failed to delete."
                else:
                    message = f"Deleted {deleted_count} files."
                dialog = CustomDialog(self, title="Done", message=message, buttons=["OK"])
                dialog.exec()
                
                # Refresh display to show updated results
                if self.file_groups:
                    # Re-apply sorting to updated groups
                    current_sort = self.sort_combo.currentText()
                    self.file_groups = self.engine.apply_sorting(
                        self.file_groups_raw,
                        current_sort,
                        group_by_type=False
                    )
                    # Update status
                    self._results_summary = self._build_results_summary(self.file_groups)
                    self.status_label.setText(self._results_summary)
                    self.status_bar.setVisible(True)
                    # Refresh current page
                    self._refresh_display()
                else:
                    # No more duplicate groups
                    self._results_summary = ""
                    self.status_label.setVisible(False)
                    self.status_bar.setVisible(False)
                    self.center_progress_container.setVisible(True)
                    self.center_progress_bar.setVisible(False)
                    self.center_status_label.setText(self._get_instruction_text())
                    if hasattr(self, "footer"):
                        self.footer.setVisible(False)
                    self._group_widgets.clear()
                    while self.results_layout.count() > 1:
                        item = self.results_layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                    self.page_combo.blockSignals(True)
                    self.page_combo.set_total_pages(1)  # Set total pages for display
                    self.page_combo.clear()
                    self.page_combo.addItem("1")
                    self.page_combo.setCurrentIndex(0)
                    self.page_combo.blockSignals(False)
                    self.prev_btn.setEnabled(False)
                    self.next_btn.setEnabled(False)
                    self.file_groups = None
                    self.file_groups_raw = None
            except ImportError:
                dialog = CustomDialog(self, title="Error", message="send2trash not available", buttons=["OK"])
                dialog.exec()


def main():
    """Main entry point."""
    _dbg = os.environ.get("CLONEWIPER_DEBUG", "").strip().lower()
    if _dbg in ("1", "true", "yes", "on"):
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    if not PYSIDE6_AVAILABLE:
        print("PySide6 not available. Please install: pip install PySide6")
        return 1
    
    print("Starting CloneWiper (Material 3)...")
    if sys.platform == 'win32':
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('CloneWiper.App.1')
        except Exception:
            pass
    app = QApplication(sys.argv)
    app_icon = _load_app_icon()
    if app_icon is not None:
        app.setWindowIcon(app_icon)
    app.setStyle("Fusion")  # Modern look
    
    # Material Design 3 global styles
    app.setStyleSheet(f"""
        QWidget {{
            font-family: 'Roboto', 'Segoe UI', sans-serif;
        }}
        QMessageBox {{
            background-color: {MD3_COLORS['bg_subtle']};
            color: {MD3_COLORS['on_surface']};
            border: 2px solid {MD3_COLORS['outline']};
            border-radius: 12px;
        }}
        QMessageBox QLabel {{
            color: {MD3_COLORS['on_surface']};
            font-family: 'Roboto', 'Segoe UI', sans-serif;
            font-size: 14px;
        }}
        QMessageBox QLabel[objectName="qt_msgbox_label"] {{
            font-size: 18px;
            font-weight: 600;
            color: {MD3_COLORS['on_surface']};
            text-align: center;
        }}
        QMessageBox QLabel[objectName="qt_msgbox_informativelabel"] {{
            font-size: 13px;
            color: {MD3_COLORS['on_surface_variant']};
            text-align: center;
        }}
        QMessageBox QPushButton {{
            background-color: {MD3_COLORS['primary']};
            color: {MD3_COLORS['on_primary']};
            border: none;
            border-radius: 20px;
            padding: 8px 24px;
            font-family: 'Roboto', 'Segoe UI', sans-serif;
            font-size: 13px;
            font-weight: 600;
            min-width: 80px;
        }}
        QMessageBox QPushButtonBox {{
            alignment: center;
        }}
        QMessageBox QPushButton:hover {{
            background-color: {MD3_COLORS['primary_container']};
        }}
        QMessageBox QPushButton:pressed {{
            background-color: {MD3_COLORS['on_primary_container']};
        }}
        QMessageBox QLabel[objectName="qt_msgboxex_icon_label"] {{
            width: 0px;
            height: 0px;
            max-width: 0px;
            max-height: 0px;
        }}
    """)
    
    window = CloneWiperApp()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())



