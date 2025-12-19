"""
CloneWiper - Qt UI Implementation

Modern UI implementation using PySide6 with Material Design 3 styling.
Qt UI provides excellent performance when handling large result sets and supports rich interactive features.

Key Features:
    - Material Design 3 styled interface
    - Masonry layout for thumbnail display
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
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import time
import threading

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QCheckBox, QComboBox, QScrollArea,
        QFrame, QListView, QStyledItemDelegate, QStyleOptionViewItem,
        QSizePolicy, QMessageBox, QFileDialog, QProgressBar, QGridLayout, QScrollBar,
        QListWidget, QDialogButtonBox, QDialog, QStyleOptionComboBox, QStyle
    )
    from PySide6.QtCore import (
        Qt, QSize, QThread, QThreadPool, QRunnable, Signal, QObject, QModelIndex,
        QAbstractListModel, QRect, QPoint, QTimer, QMutex, QWaitCondition, Slot, QSettings,
        QVariantAnimation, QEasingCurve, QPropertyAnimation, QEvent
    )
    from PySide6.QtGui import (
        QPixmap, QPainter, QFont, QColor, QPen, QBrush, QImage, QIcon, QPainterPath
    )
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    print("Warning: PySide6 not found. Please install it with 'pip install PySide6'")

from core.engine import ScanEngine, FileItem, Group

if not PYSIDE6_AVAILABLE:
    print("Qt UI requires PySide6. Please install it with: pip install PySide6")
    sys.exit(1)



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
        
        # Load favicon - try multiple paths (works in dev and when packaged)
        # Get the directory where the script is located
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        icon_paths = [
            os.path.join(base_path, "icons", "favicon.ico"),
            os.path.join(base_path, "favicon.ico"),  # Fallback for old builds
            os.path.join(os.getcwd(), "icons", "favicon.ico"),
            os.path.join(os.getcwd(), "favicon.ico"),  # Fallback
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "favicon.ico"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.ico"),  # Fallback
            "icons/favicon.ico",
            "favicon.ico"  # Fallback
        ]
        
        icon_loaded = False
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                try:
                    icon = QIcon(icon_path)
                    pixmap = icon.pixmap(28, 28)
                    if not pixmap.isNull():
                        self.icon_label.setPixmap(pixmap)
                        icon_loaded = True
                        break
                except Exception:
                    continue
        
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
            self.max_btn.setText("⬜")
        else:
            self.parent.showMaximized()
            self.max_btn.setText("❐")
        self._is_maximized = not self._is_maximized

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
    
    class Signals(QObject):
        thumb_ready = Signal(str, QPixmap, float)  # path, pixmap, aspect_ratio
        meta_ready = Signal(str, str)  # path, metadata_text
        error = Signal(str, str)  # path, error_msg
        
        def __init__(self, parent=None):
            super().__init__(parent)
    
    def __init__(self, file_path: str, max_width: int, page_token: float):
        super().__init__()
        self.file_path = file_path
        self.max_width = max_width
        self.page_token = page_token
        # Create signals object - must be created in main thread for proper signal/slot connection
        # The signals object will be moved to main thread in _request_thumbnail
        self.signals = ThumbnailWorker.Signals()
    
    def run(self):
        """Load thumbnail in background thread."""
        try:
            print(f"DEBUG: ThumbnailWorker.run() called for {os.path.basename(self.file_path)}")
            ext = os.path.splitext(self.file_path)[1].lower()
            
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
                    from PIL import Image
                    with Image.open(self.file_path) as img:
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
                        print(f"DEBUG: ThumbnailWorker emitted thumb_ready for {os.path.basename(self.file_path)}")
                        
                        # Metadata
                        meta = f"{width}x{height}"
                        self.signals.meta_ready.emit(self.file_path, meta)
                except Exception as e:
                    print(f"DEBUG: ThumbnailWorker error for {os.path.basename(self.file_path)}: {e}")
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
                    print(f"DEBUG: ThumbnailWorker RAW error for {os.path.basename(self.file_path)}: {e}")
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
                        
                        # Metadata
                        cap2 = cv2.VideoCapture(self.file_path)
                        w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap2.get(cv2.CAP_PROP_FPS)
                        duration_sec = 0
                        if fps > 0:
                            frames = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
                            duration_sec = int(frames / fps)
                        cap2.release()
                        dur_str = f"{duration_sec//60}:{duration_sec%60:02d}"
                        meta = f"{w}x{h} • {dur_str}"
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
                            import pypdfium2 as pdfium
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
                            
                            page = pdf[0]
                            bitmap = page.render(scale=2, rotation=0)
                            pil_image = bitmap.to_pil()
                            width, height = pil_image.size
                            aspect = width / height
                            base_width = min(self.max_width, 512)
                            scale = base_width / width
                            new_height = int(height * scale)
                            img_resized = pil_image.resize((base_width, new_height), Image.Resampling.LANCZOS)
                            img_bytes = img_resized.tobytes('raw', 'RGB')
                            qimg = QImage(img_bytes, base_width, new_height, QImage.Format.Format_RGB888)
                            pixmap = QPixmap.fromImage(qimg)
                            self.signals.thumb_ready.emit(self.file_path, pixmap, aspect)
                            self.signals.meta_ready.emit(self.file_path, f"{len(pdf)} pages")
                            pdf.close()
                            return
                        except ImportError as e:
                            pdfium_error = f"pypdfium2 not installed: {e}"
                            print(f"DEBUG: pypdfium2 import failed: {e}")
                        except Exception as e:
                            import traceback
                            pdfium_error = f"pypdfium2 error: {e}"
                            print(f"DEBUG: pypdfium2 processing failed for {os.path.basename(self.file_path)}: {e}")
                            print(f"DEBUG: Traceback: {traceback.format_exc()}")

                    # DOCUMENT FALLBACK/PRIMARY: PyMuPDF (fitz)
                    try:
                        import fitz  # PyMuPDF
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
                        
                        page = doc.load_page(0)
                        pix = page.get_pixmap(alpha=False)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
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
                        
                        meta = "Ebook"
                        if ext == '.pdf': meta = f"{doc.page_count} pages"
                        elif ext == '.epub': meta = "Ebook (EPUB)"
                        elif ext == '.mobi': meta = "Ebook (MOBI)"
                        elif ext == '.azw3': meta = "Ebook (AZW3)"
                        self.signals.meta_ready.emit(self.file_path, meta)
                        doc.close()
                        return
                    except ImportError as e:
                        pymupdf_error = f"PyMuPDF not installed: {e}"
                        print(f"DEBUG: PyMuPDF import failed: {e}")
                    except Exception as e:
                        import traceback
                        pymupdf_error = f"PyMuPDF error: {e}"
                        print(f"DEBUG: PyMuPDF processing failed for {os.path.basename(self.file_path)}: {e}")
                        print(f"DEBUG: Traceback: {traceback.format_exc()}")

                    # EPUB SPECIAL: Deep Manifest Parsing (Last Resort if fitz fails)
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
                        except Exception as epub_error:
                            # If EPUB special parsing fails, just pass (will be caught by outer exception handler)
                            pass
                    
                    # If both PDF libraries failed, emit error with helpful message
                    if ext == '.pdf' and pdfium_error and pymupdf_error:
                        import platform
                        system = platform.system()
                        install_cmd = "pip3" if system == "Darwin" else "pip"
                        error_msg = f"PDF thumbnail failed. Both libraries failed:\n  pypdfium2: {pdfium_error}\n  PyMuPDF: {pymupdf_error}\n\nInstall one of them:\n  {install_cmd} install pypdfium2\n  or\n  {install_cmd} install PyMuPDF"
                        print(f"DEBUG: PDF thumbnail error for {os.path.basename(self.file_path)}: {error_msg}")
                        self.signals.error.emit(self.file_path, error_msg)
                    elif ext in {'.epub', '.mobi', '.azw3'} and pymupdf_error:
                        import platform
                        system = platform.system()
                        install_cmd = "pip3" if system == "Darwin" else "pip"
                        error_msg = f"Ebook thumbnail failed. Install PyMuPDF:\n  {install_cmd} install PyMuPDF\n\nError: {pymupdf_error}"
                        print(f"DEBUG: Ebook thumbnail error for {os.path.basename(self.file_path)}: {error_msg}")
                        self.signals.error.emit(self.file_path, error_msg)

                except Exception as e:
                    import traceback
                    error_msg = f"Doc error: {e}"
                    print(f"DEBUG: PDF/EPUB thumbnail error for {os.path.basename(self.file_path)}: {error_msg}")
                    print(traceback.format_exc())
                    self.signals.error.emit(self.file_path, error_msg)
            
            # Audio thumbnail (Album Art via Mutagen)
            elif ext in {'.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus', '.alac', '.ape', '.aiff'}:
                try:
                    try:
                        import mutagen
                        from mutagen.mp3 import MP3
                        from mutagen.flac import FLAC
                        from mutagen.mp4 import MP4
                        from mutagen.oggvorbis import OggVorbis
                        from mutagen.oggopus import OggOpus
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
                        except Exception as img_error:
                            print(f"DEBUG: Failed to process album art image for {os.path.basename(self.file_path)}: {img_error}")
                    
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
                            print(f"DEBUG: Failed to extract tags for {os.path.basename(self.file_path)}: {tag_error}")
                    
                    meta = f"{artist} - {album} • {dur_str}"
                    self.signals.meta_ready.emit(self.file_path, meta)
                    
                except Exception as e:
                    print(f"DEBUG: Audio thumbnail error for {os.path.basename(self.file_path)}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.signals.error.emit(self.file_path, f"Audio error: {e}")
        
        except Exception as e:
            self.signals.error.emit(self.file_path, str(e))


class FileCardDelegate(QStyledItemDelegate):
    """Custom delegate for rendering file cards with thumbnails (masonry layout)."""
    
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
        """Return variable height based on aspect ratio (masonry effect)."""
        path = index.data(Qt.ItemDataRole.UserRole)
        if not path:
            return QSize(self.card_width, 200)
        
        self.mutex.lock()
        try:
            aspect = self.aspect_ratios.get(path)
            if aspect is None:
                # Try to infer from file type or use default
                ext = os.path.splitext(path)[1].lower()
                if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}:
                    aspect = 1.0  # Default square-ish
                elif ext in {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}:
                    aspect = 16.0 / 9.0  # Common video aspect
                else:
                    aspect = 1.0
                self._default_aspects[path] = aspect
            else:
                self._default_aspects.pop(path, None)
        finally:
            self.mutex.unlock()
        
        # Calculate height based on aspect ratio
        thumb_height = int(self.card_width / aspect) if aspect > 0 else self.card_width
        thumb_height = max(150, min(thumb_height, self.card_width * 3))
        
        # Add space for text/metadata (Material 3 spacing)
        text_height = 90
        total_height = thumb_height + text_height
        
        return QSize(self.card_width, total_height)
    
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Paint file card with thumbnail, metadata, and selection state."""
        path = index.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        
        rect = option.rect
        painter.save()
        
        # Selection state
        self.mutex.lock()
        try:
            selected = self.selection_state.get(path, False)
            thumb = self.thumbnails.get(path)
            aspect = self.aspect_ratios.get(path, 1.0)
            meta = self.metadata.get(path, "")
        finally:
            self.mutex.unlock()
        
        # Background (Material 3 Card)
        border_radius = 12
        if selected:
            # Material 3 error container color with rounded corners
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(MD3_COLORS['error_container'])))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), border_radius, border_radius)
            # Red border
            pen = QPen(QColor(MD3_COLORS['error']), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), border_radius, border_radius)
        else:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(MD3_COLORS['surface'])))  # Material 3 surface
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), border_radius, border_radius)
            # Subtle border
            pen = QPen(QColor(MD3_COLORS['surface_variant']), 1)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), border_radius, border_radius)
        
        # Thumbnail (Material 3 rounded corners)
        # Use consistent padding (8px) for all sides
        padding = 8
        # Thumbnail corner radius = card radius - padding (12 - 8 = 4px)
        thumb_radius = border_radius - padding
        thumb_height = int(self.card_width / aspect) if aspect > 0 else self.card_width
        thumb_height = max(150, min(thumb_height, self.card_width * 3))
        thumb_rect = QRect(rect.x() + padding, rect.y() + padding, self.card_width - (padding * 2), thumb_height)
        
        if thumb:
            # Create rounded thumbnail with full visibility
            scaled = thumb.scaled(thumb_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            thumb_x = thumb_rect.x() + (thumb_rect.width() - scaled.width()) // 2
            # Align thumbnail to top of thumb_rect (no extra vertical spacing)
            thumb_y = thumb_rect.y()
            
            # Draw with rounded corners (card radius - padding)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rounded_rect = QRect(thumb_x, thumb_y, scaled.width(), scaled.height())
            
            # Create a rounded rectangle path for clipping
            clip_path = QPainterPath()
            clip_path.addRoundedRect(rounded_rect, thumb_radius, thumb_radius)
            painter.setClipPath(clip_path)
            
            # Draw pixmap with rounded corners
            painter.drawPixmap(rounded_rect, scaled)
            painter.setClipping(False)
            
            # Draw subtle border around thumbnail
            painter.setPen(QPen(QColor(0, 0, 0, 20), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rounded_rect, thumb_radius, thumb_radius)
        else:
            # Placeholder (Material 3 surface variant)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(MD3_COLORS['surface_variant'])))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(thumb_rect, thumb_radius, thumb_radius)
            
            painter.setPen(QColor(MD3_COLORS['on_surface_variant']))
            font = QFont("Roboto", 10)
            painter.setFont(font)
            painter.drawText(thumb_rect, Qt.AlignmentFlag.AlignCenter, "Loading...")
        
        # Text/metadata (Material 3 Typography)
        # Use same padding (8px) for text area to align with thumbnail padding
        text_y = thumb_rect.y() + thumb_height + padding
        text_rect = QRect(rect.x() + padding, text_y, rect.width() - (padding * 2), rect.height() - (thumb_rect.y() + thumb_height + padding))
        
        name = os.path.basename(path)
        try:
            size_str = self._format_size(os.path.getsize(path))
            mtime_str = time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(path)))
        except Exception:
            size_str = ""
            mtime_str = ""
        
        parts = [p for p in [size_str, meta, mtime_str] if p]
        meta_text = " • ".join(parts)
        
        # Filename (Material 3 Title Medium) with scrolling
        painter.setPen(QColor(MD3_COLORS['on_surface']))  # Material 3 on-surface
        font = QFont("Roboto", 14, QFont.Weight.Medium)
        painter.setFont(font)
        name_rect = QRect(text_rect.x(), text_rect.y(), text_rect.width(), 22)
        
        # Calculate text width (always calculate, not just when hovering)
        fm = painter.fontMetrics()
        name_width = fm.horizontalAdvance(name)
        is_name_overflow = name_width > name_rect.width()
        
        # Store text width for scrolling calculation (always store)
        existing_dir_width = self._text_widths.get(path, (0, 0))[1]
        self._text_widths[path] = (name_width, existing_dir_width)
        
        # Get scroll offset if hovering
        name_offset = 0
        if is_name_overflow and path == self._hovered_path:
            name_offset = self._name_scroll_offset.get(path, 0)
            max_offset = max(0, name_width - name_rect.width() + 20)  # Add padding
            if name_offset > max_offset:
                name_offset = max_offset
            elif name_offset < 0:
                name_offset = 0
        
        # Draw filename with clipping and scrolling
        painter.save()
        painter.setClipRect(name_rect)
        if is_name_overflow and path == self._hovered_path:
            painter.drawText(name_rect.adjusted(-int(name_offset), 0, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, name)
        else:
            painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, name)
        painter.restore()
        
        # Metadata (Material 3 Body Small)
        if meta_text:
            meta_rect = QRect(text_rect.x(), text_rect.y() + 28, text_rect.width(), 22)
            painter.setPen(QColor(MD3_COLORS['on_surface_variant']))  # Material 3 on-surface-variant
            font_body = QFont("Roboto", 12)
            painter.setFont(font_body)
            painter.drawText(meta_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, meta_text)
        
        # Directory (Material 3 Label Small) with scrolling
        dir_path = os.path.dirname(path)
        if dir_path:
            dir_rect = QRect(text_rect.x(), text_rect.y() + 52, text_rect.width(), 18)
            painter.setPen(QColor(MD3_COLORS['on_surface_variant']))
            font_label = QFont("Roboto", 11)
            painter.setFont(font_label)
            
            # Calculate text width (always calculate, not just when hovering)
            fm = painter.fontMetrics()
            dir_width = fm.horizontalAdvance(dir_path)
            is_dir_overflow = dir_width > dir_rect.width()
            
            # Store text width for scrolling calculation (always store)
            existing_name_width = self._text_widths.get(path, (0, 0))[0]
            self._text_widths[path] = (existing_name_width, dir_width)
            
            # Get scroll offset if hovering
            dir_offset = 0
            if is_dir_overflow and path == self._hovered_path:
                dir_offset = self._dir_scroll_offset.get(path, 0)
                max_offset = max(0, dir_width - dir_rect.width() + 20)  # Add padding
                if dir_offset > max_offset:
                    dir_offset = max_offset
                elif dir_offset < 0:
                    dir_offset = 0
            
            # Draw directory path with clipping and scrolling
            painter.save()
            painter.setClipRect(dir_rect)
            if is_dir_overflow and path == self._hovered_path:
                painter.drawText(dir_rect.adjusted(-int(dir_offset), 0, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, dir_path)
            else:
                painter.drawText(dir_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, dir_path)
            painter.restore()
        
        painter.restore()
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def set_hovered_path(self, path: Optional[str], view):
        """Set the currently hovered path and start/stop scrolling animation.
        
        Logic:
        1. When name is completely displayed and user still mouse over: 
           Keep current scroll position, don't reset offset, don't start timer
        2. When mouse leaves: 
           Reset scroll offset to 0, trigger repaint to show beginning
        3. When name is not completely displayed and user hovers:
           Reset offset to 0, start scrolling animation from beginning
        """
        if path == self._hovered_path:
            return
        
        # Save previous path before changing _hovered_path
        prev_path = self._hovered_path
        
        # Stop previous scrolling timer (but keep offsets for now)
        if prev_path and prev_path in self._scroll_timers:
            timer = self._scroll_timers.pop(prev_path)
            timer.stop()
            timer.deleteLater()
        
        # Update hovered path
        self._hovered_path = path
        
        if path is None:
            # Case 2: Mouse left - reset scroll positions for the previous path
            if prev_path:
                # Always reset scroll offsets to 0 when mouse leaves
                self._name_scroll_offset[prev_path] = 0
                self._dir_scroll_offset[prev_path] = 0
                
                # Trigger repaint to reset visual position and show beginning of filename
                model = view.model()
                if model:
                    try:
                        for i in range(model.rowCount()):
                            idx = model.index(i)
                            if idx.data(Qt.ItemDataRole.UserRole) == prev_path:
                                view.update(idx)
                                break
                    except (AttributeError, RuntimeError):
                        pass
            return
        
        # Case 1 & 3: Mouse hovering over a path
        model = view.model()
        if model is None:
            return
        
        # Get text widths and display area width
        # name_rect.width() = card_width - 24 (accounting for padding on both sides)
        name_width, dir_width = self._text_widths.get(path, (0, 0))
        display_width = self.card_width - 24  # Actual display area width (name_rect.width())
        
        # Check if offsets already exist (might be from previous scroll)
        has_existing_offset = path in self._name_scroll_offset or path in self._dir_scroll_offset
        
        # Case 1: Text widths are known - check if text is completely displayed
        # Text is completely displayed when: name_width <= display_width
        # This means the last character is within the visible area (no scrolling needed)
        if name_width > 0 and dir_width > 0:
            name_fits = name_width <= display_width
            dir_fits = dir_width <= display_width
            
            if name_fits and dir_fits:
                # Both texts fit completely - keep current position, don't start timer
                # Don't reset offsets - they might be from a previous scroll to the end
                return
        
        # Case 3: Text needs scrolling or widths not calculated yet
        # If text widths are not calculated yet and offsets exist, don't reset them
        # Wait for width calculation in _update_scroll to decide if scrolling is needed
        if name_width > 0 and dir_width > 0:
            # Text widths are known - text needs scrolling
            # Only reset offsets if this is first time hovering (no offset exists)
            if path not in self._name_scroll_offset:
                self._name_scroll_offset[path] = 0
            if path not in self._dir_scroll_offset:
                self._dir_scroll_offset[path] = 0
        else:
            # Text widths not calculated yet
            # If offsets exist, don't reset them - text might be completely displayed
            # Only initialize if first time hovering
            if not has_existing_offset:
                self._name_scroll_offset[path] = 0
                self._dir_scroll_offset[path] = 0
            # If offsets exist, keep them and let _update_scroll decide after width calculation
        
        # Trigger initial repaint to calculate text widths (if not already calculated)
        try:
            for i in range(model.rowCount()):
                idx = model.index(i)
                if idx.data(Qt.ItemDataRole.UserRole) == path:
                    view.update(idx)
                    break
        except (AttributeError, RuntimeError):
            return
        
        # Start scrolling animation timer
        # _update_scroll will stop it if text is completely displayed
        timer = QTimer()
        timer.setSingleShot(False)
        timer.timeout.connect(lambda: self._update_scroll(path, view))
        timer.start(50)  # Update every 50ms
        self._scroll_timers[path] = timer
    
    def _update_scroll(self, path: str, view):
        """Update scroll position for hovered text.
        
        This is called by the timer when scrolling is active.
        It handles:
        - Waiting for text width calculation
        - Stopping timer if text is completely displayed (keep current position)
        - Continuing scroll animation if text needs scrolling
        """
        # Safety check
        if path != self._hovered_path:
            return
        
        model = view.model()
        if model is None:
            return
        
        # Get text widths and display area width
        # name_rect.width() = card_width - 24 (accounting for padding on both sides)
        name_width, dir_width = self._text_widths.get(path, (0, 0))
        display_width = self.card_width - 24  # Actual display area width (name_rect.width())
        
        # Wait for text width calculation
        if name_width == 0 and dir_width == 0:
            # Trigger repaint to calculate widths
            try:
                for i in range(model.rowCount()):
                    idx = model.index(i)
                    if idx.data(Qt.ItemDataRole.UserRole) == path:
                        view.update(idx)
                        break
            except (AttributeError, RuntimeError):
                pass
            return
        
        # Check if scrolling is needed
        # Text is completely displayed when: name_width <= display_width
        # This means the last character is within the visible area (no scrolling needed)
        name_fits = name_width <= display_width
        dir_fits = dir_width <= display_width
        
        # If text is completely displayed, stop timer but keep current position
        if name_fits and dir_fits:
            if path in self._scroll_timers:
                timer = self._scroll_timers.pop(path)
                timer.stop()
                timer.deleteLater()
            # Don't reset offsets - keep current position
            return
        
        # Continue scrolling animation
        needs_update = False
        should_stop_timer = False
        
        # Update name scroll
        # name_needs_scroll means name_width > display_width
        if not name_fits:
            current_offset = self._name_scroll_offset.get(path, 0)
            # When the last character is visible: current_offset >= (name_width - display_width)
            # At this point, the last character is at or before the right edge of visible area
            min_offset_to_show_end = name_width - display_width
            
            if current_offset >= min_offset_to_show_end:
                # Last character is already visible - stop scrolling for name, keep current position
                # Don't reset to 0, just stop updating
                pass  # Keep current offset
            else:
                # Scroll forward until last character is visible
                self._name_scroll_offset[path] = current_offset + 2
                needs_update = True
        
        # Update dir scroll
        # dir_needs_scroll means dir_width > display_width
        if not dir_fits:
            current_offset = self._dir_scroll_offset.get(path, 0)
            # When the last character is visible: current_offset >= (dir_width - display_width)
            min_offset_to_show_end = dir_width - display_width
            
            if current_offset >= min_offset_to_show_end:
                # Last character is already visible - stop scrolling for dir, keep current position
                # Don't reset to 0, just stop updating
                pass  # Keep current offset
            else:
                # Scroll forward until last character is visible
                self._dir_scroll_offset[path] = current_offset + 2
                needs_update = True
        
        # Check if both texts have reached the end (last character visible)
        # If so, stop the timer but keep current positions
        name_at_end = name_fits or (not name_fits and self._name_scroll_offset.get(path, 0) >= (name_width - display_width))
        dir_at_end = dir_fits or (not dir_fits and self._dir_scroll_offset.get(path, 0) >= (dir_width - display_width))
        
        if name_at_end and dir_at_end:
            # Both texts have reached the end - stop timer, keep positions
            if path in self._scroll_timers:
                timer = self._scroll_timers.pop(path)
                timer.stop()
                timer.deleteLater()
            return
        
        # Trigger repaint if scroll position changed
        if needs_update:
            try:
                for i in range(model.rowCount()):
                    idx = model.index(i)
                    if idx.data(Qt.ItemDataRole.UserRole) == path:
                        view.update(idx)
                        break
            except (AttributeError, RuntimeError):
                pass


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
        
        return None
    
    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class GroupWidget(QFrame):
    """Collapsible widget for a duplicate group with auto-expanding masonry layout."""
    
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
        header = QFrame()
        header.setFixedHeight(48)
        header.setStyleSheet(f"background-color: {MD3_COLORS['bg_subtle']}; border: none; border-top-left-radius: 12px; border-top-right-radius: 12px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 4, 16, 4)
        header_layout.setSpacing(12)
        
        self.toggle_btn = QPushButton("▼")
        self.toggle_btn.setFixedSize(32, 32)
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
        
        total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
        size_str = self._format_size(total_size)
        # Use group index instead of hash ID
        if total_groups > 0:
            title = QLabel(f"Group {group_index + 1}/{total_groups} • {len(files)} files • {size_str}")
        else:
            title = QLabel(f"{len(files)} files • {size_str}")
        title.setStyleSheet(f"color: {MD3_COLORS['on_surface']}; font-size: 14px; font-weight: 500; border: none;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # List view with masonry delegate
        self.list_view = QListView()
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setSpacing(12)
        self.list_view.setUniformItemSizes(False)
        self.list_view.setFlow(QListView.Flow.LeftToRight)
        self.list_view.setWrapping(True)
        self.list_view.setGridSize(QSize())
        self.list_view.setMovement(QListView.Movement.Static)
        self.list_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_view.setFrameShape(QFrame.Shape.NoFrame)
        self.list_view.setStyleSheet("background-color: transparent;")
        
        self.delegate = FileCardDelegate()
        self.delegate.set_card_width(320)
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
        """Calculate and set the height of the list_view based on its contents."""
        if not self.is_expanded:
            self.list_view.setFixedHeight(0)
            return
            
        # Small delay to let Qt's layout engine settle
        QTimer.singleShot(10, self._do_update_height)

    def _do_update_height(self):
        if not self.isVisible():
            return
            
        count = self.model.rowCount()
        if count == 0:
            self.list_view.setFixedHeight(0)
            return
            
        # Find maximum Y + height of any item
        max_bottom = 0
        for i in range(count):
            rect = self.list_view.visualRect(self.model.index(i))
            max_bottom = max(max_bottom, rect.bottom())
            
        # Add some padding
        final_height = max_bottom + 16
        self.list_view.setFixedHeight(max(100, final_height))
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
            self.toggle_btn.setText("▼")
        else:
            self.list_view.hide()
            self.toggle_btn.setText("▶")
    
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
        # Update view - force size hint recalculation and layout update
        try:
            for i in range(self.model.rowCount()):
                idx = self.model.index(i)
                if idx.data(Qt.ItemDataRole.UserRole) == path:
                    self.list_view.update(idx)
                    self.model.dataChanged.emit(idx, idx, [])
                    self.list_view.updateGeometry()
                    self.list_view.doItemsLayout() 
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
        
        # Default buttons
        if buttons is None:
            buttons = ["OK"]
        
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {MD3_COLORS['bg_subtle']};
                border: 2px solid {MD3_COLORS['outline']};
                border-radius: 12px;
            }}
        """)
        
        # Layout
        layout = QVBoxLayout(self)
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


class CloneWiperApp(QMainWindow):
    """Main CloneWiper application window."""
    
    # Signals for thread-safe UI updates
    progress_updated = Signal(float)
    status_updated = Signal(str)
    results_ready = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CloneWiper - Smart Duplicate Finder")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setGeometry(100, 100, 1400, 850)
        
        # Set window icon (for taskbar) - try multiple paths (works in dev and when packaged)
        # Get the directory where the script is located
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        icon_paths = [
            os.path.join(base_path, "icons", "favicon.ico"),
            os.path.join(base_path, "favicon.ico"),  # Fallback for old builds
            os.path.join(os.getcwd(), "icons", "favicon.ico"),
            os.path.join(os.getcwd(), "favicon.ico"),  # Fallback
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "favicon.ico"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.ico"),  # Fallback
            "icons/favicon.ico",
            "favicon.ico"  # Fallback
        ]
        
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                try:
                    icon = QIcon(icon_path)
                    if not icon.isNull():
                        self.setWindowIcon(icon)
                        break
                except Exception:
                    continue
        
        # Resize state
        self._resize_edge_active = False
        self._resize_start_pos = None
        self._resize_start_geom = None
        self.setMouseTracking(True)
        
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
        self.groups_per_page = 50
        self._scroll_anchor: Optional[Tuple[str, int]] = None  # (group_id, scroll_value) for refresh restore
        
        # Thumbnail cache
        self.thumb_cache: Dict[str, QPixmap] = OrderedDict()
        self.thumb_cache_max = 450
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
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet(f"background-color: {MD3_COLORS['surface']};")
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
        control_layout.setContentsMargins(24, 0, 24, 0)
        control_layout.setSpacing(8)
        
        # Path selection area (Condensed)
        path_container = QFrame()
        path_container.setStyleSheet("background-color: transparent;")
        path_hbox = QHBoxLayout(path_container)
        path_hbox.setContentsMargins(0, 0, 0, 0)
        path_hbox.setSpacing(12)
        
        path_label = QLabel("Scan Folders:")
        path_label.setStyleSheet(f"color: {MD3_COLORS['on_surface_variant']}; font-size: 13px; font-weight: 500;")
        path_hbox.addWidget(path_label)
        
        # Path List Widget (With Scrollbar)
        self.path_list_widget = QListWidget()
        self.path_list_widget.setMinimumWidth(300)
        self.path_list_widget.setFixedHeight(36) # Default height
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
                padding: 1px;
            }}
            QListWidget::item {{
                padding: 4px 8px;
                border-radius: 6px;
                margin: 1px 2px;
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
        
        # Folder removal via Context Menu and Delete Key
        self.path_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.path_list_widget.customContextMenuRequested.connect(self._show_path_context_menu)
        self.path_list_widget.keyPressEvent = self._path_list_key_press
        
        path_hbox.addWidget(self.path_list_widget)
        
        # Add Folder Button (Icon Only or Small)
        add_btn = QPushButton("+ Add Folder")
        add_btn.setFixedSize(110, 32)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {MD3_COLORS['primary_container']};
                color: {MD3_COLORS['on_primary_container']};
                font-size: 12px;
                font-weight: 500;
                border: none;
                border-radius: 16px;
            }}
            QPushButton:hover {{ background-color: #d0bcff; }}
        """)
        add_btn.clicked.connect(self._browse_path)
        path_hbox.addWidget(add_btn)
        
        control_layout.addWidget(path_container)
        
        # Add stretch to push controls to the right
        control_layout.addStretch()
        
        # PHash Toggle
        self.phash_check = QCheckBox("Multi-Algorithm Perceptual Hash")
        self.phash_check.setStyleSheet(f"""
            QCheckBox {{
                color: {MD3_COLORS['on_surface']};
                background-color: transparent;
                font-size: 13px;
                spacing: 8px;
                padding: 4px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {MD3_COLORS['outline']};
                border-radius: 4px;
                background-color: {MD3_COLORS['bg_subtle']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {MD3_COLORS['primary']};
                border-color: {MD3_COLORS['primary']};
            }}
        """)
        control_layout.addWidget(self.phash_check)
        
        # Scan Button (Filled)
        self.scan_btn = QPushButton("Start Scanning")
        self.scan_btn.setFixedSize(160, 40)
        self.scan_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {MD3_COLORS['primary']};
                color: {MD3_COLORS['on_primary']};
                font-size: 13px;
                font-weight: 600;
                border-radius: 20px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{ background-color: {MD3_COLORS['primary_container']}; }}
            QPushButton:disabled {{ background-color: {MD3_COLORS['surface_variant']}; color: {MD3_COLORS['on_surface_variant']}; }}
        """)
        self.scan_btn.clicked.connect(self._start_scanning)
        control_layout.addWidget(self.scan_btn)
        self.main_layout.addWidget(control_panel)
        
        # Status bar (Material 3)
        status_bar = QFrame()
        status_bar.setFixedHeight(30) # Compact status bar
        status_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {MD3_COLORS['bg_tertiary']};
                border: none;
            }}
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(24, 0, 24, 0)
        status_layout.setSpacing(16)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"""
            color: {MD3_COLORS['on_surface_variant']};
            font-size: 14px;
            font-family: 'Roboto', 'Segoe UI', sans-serif;
            padding: 0px;
        """)
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8) # More prominent
        self.progress_bar.setMinimumWidth(400)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {MD3_COLORS['surface_variant']};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {MD3_COLORS['primary']};
                border-radius: 4px;
            }}
        """)
        status_layout.addWidget(self.progress_bar)
        
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
        
        self.results_container = QWidget()
        self.results_container.setStyleSheet(f"background-color: {MD3_COLORS['bg_tertiary']};")
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(24, 16, 24, 16)
        self.results_layout.setSpacing(16)
        self.results_layout.addStretch()
        
        self.scroll_area.setWidget(self.results_container)
        self.main_layout.addWidget(self.scroll_area)
        
        # KEY: Apply stretch factors to prevent top widgets from vertical expansion
        self.main_layout.setStretch(0, 0) # Custom Title Bar
        self.main_layout.setStretch(1, 0) # Control Panel
        self.main_layout.setStretch(2, 0) # Status Bar
        self.main_layout.setStretch(3, 1) # Results Area (Grow)
        
        # Footer (Material 3 Bottom App Bar style)
        footer = QFrame()
        footer.setFixedHeight(64)
        footer.setFrameShape(QFrame.Shape.NoFrame)
        footer.setStyleSheet(f"""
            QFrame {{
                background-color: {MD3_COLORS['bg_tertiary']};
                border: none;
            }}
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 8, 24, 8)
        footer_layout.setSpacing(12)
        footer_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)  # Vertical center alignment
        
        # Quick select buttons (Material 3 Outlined Button)
        outlined_button_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {MD3_COLORS['primary']};
                font-size: 13px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 24px;
                border: 1px solid {MD3_COLORS['outline']};
                border-radius: 20px;
                min-width: 100px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: rgba(103, 80, 164, 0.08);
                border-color: {MD3_COLORS['primary']};
            }}
            QPushButton:pressed {{
                background-color: rgba(103, 80, 164, 0.12);
            }}
        """
        
        keep_newest_btn = QPushButton("Keep Newest")
        keep_newest_btn.setFixedHeight(40)
        keep_newest_btn.setStyleSheet(outlined_button_style)
        keep_newest_btn.clicked.connect(lambda: self._quick_select('newest'))
        footer_layout.addWidget(keep_newest_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        keep_oldest_btn = QPushButton("Keep Oldest")
        keep_oldest_btn.setFixedHeight(40)
        keep_oldest_btn.setStyleSheet(outlined_button_style)
        keep_oldest_btn.clicked.connect(lambda: self._quick_select('oldest'))
        footer_layout.addWidget(keep_oldest_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.keep_best_btn = QPushButton("Keep Best")
        self.keep_best_btn.setFixedHeight(40)
        self.keep_best_btn.setStyleSheet(outlined_button_style)
        self.keep_best_btn.clicked.connect(lambda: self._quick_select('best_res'))
        footer_layout.addWidget(self.keep_best_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.keep_raw_btn = QPushButton("Keep RAW")
        self.keep_raw_btn.setFixedHeight(40)
        self.keep_raw_btn.setStyleSheet(outlined_button_style)
        self.keep_raw_btn.clicked.connect(lambda: self._quick_select('keep_raw'))
        footer_layout.addWidget(self.keep_raw_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        # Group controls on the left
        footer_layout.addSpacing(24)
        
        # Scope toggle (Switch)
        toggle_container = QWidget()
        toggle_hbox = QHBoxLayout(toggle_container)
        toggle_hbox.setContentsMargins(0, 0, 0, 0)
        toggle_hbox.setSpacing(8)
        toggle_hbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)  # Vertical center alignment
        toggle_container.setStyleSheet("background-color: transparent;") # Ensure transparent background for container
        
        toggle_label = QLabel("All Pages")
        toggle_label.setStyleSheet(f"color: {MD3_COLORS['on_surface']}; font-size: 13px;")
        toggle_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        
        self.scope_toggle = ToggleSwitch(active_color=MD3_COLORS['primary'], bg_color=MD3_COLORS['surface_variant'])
        
        toggle_hbox.addWidget(toggle_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        toggle_hbox.addWidget(self.scope_toggle, alignment=Qt.AlignmentFlag.AlignVCenter)
        footer_layout.addWidget(toggle_container, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        footer_layout.addSpacing(16)
        
        # Sort (No Label) - Custom QComboBox with centered text
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
        # Ensure font matches exactly with the displayed text
        class CenteredComboDelegate(QStyledItemDelegate):
            def paint(self, painter, option, index):
                # Match exact font settings: 13px, weight 500 (Medium)
                font = QFont("Roboto", 13, QFont.Weight.Medium)
                font.setFamilies(["Roboto", "Segoe UI", "sans-serif"])
                painter.setFont(font)
                option.displayAlignment = Qt.AlignmentFlag.AlignCenter
                super().paint(painter, option, index)
        self.sort_combo.setItemDelegate(CenteredComboDelegate(self.sort_combo))
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
        self.prev_btn.setStyleSheet(icon_button_style)
        self.prev_btn.clicked.connect(self._prev_page)
        footer_layout.addWidget(self.prev_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.page_label = QLabel("1/1")
        self.page_label.setFixedWidth(80)
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.page_label.setStyleSheet(f"""
            color: {MD3_COLORS['on_surface']};
            font-size: 14px;
            font-weight: 500;
            font-family: 'Roboto', 'Segoe UI', sans-serif;
            padding: 0px;
            background-color: transparent;
        """)
        footer_layout.addWidget(self.page_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedSize(40, 40)
        self.next_btn.setStyleSheet(icon_button_style)
        self.next_btn.clicked.connect(self._next_page)
        footer_layout.addWidget(self.next_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        footer_layout.addStretch()
        
        # Delete button (Material 3 Error Button)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setFixedHeight(40)
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
        self.clear_select_btn = QPushButton("Clear Selection")
        self.clear_select_btn.setFixedHeight(40)
        self.clear_select_btn.setStyleSheet(outlined_button_style)
        self.clear_select_btn.setEnabled(False)
        self.clear_select_btn.clicked.connect(self._clear_selection)
        footer_layout.addWidget(self.clear_select_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.main_layout.addWidget(footer)
    
    def _adjust_path_list_height(self):
        """Dynamically adjust path list height based on items (max 1.5 rows)."""
        count = self.path_list_widget.count()
        if count <= 1:
            h = 36
        else:
            # 54px provides space for 1 full item and hints at the 2nd (1.5 rows)
            h = 54 
        self.path_list_widget.setFixedHeight(h)
        self.updateGeometry()

    def _path_list_key_press(self, event):
        """Handle Delete key for folder removal."""
        if event.key() == Qt.Key.Key_Delete:
            item = self.path_list_widget.currentItem()
            if item:
                self.path_list_widget.takeItem(self.path_list_widget.row(item))
        else:
            # Fallback to standard QListWidget key handling
            QListWidget.keyPressEvent(self.path_list_widget, event)

    def _show_path_context_menu(self, position):
        from PySide6.QtWidgets import QMenu
        menu = QMenu()
        remove_action = menu.addAction("Remove Folder")
        action = menu.exec(self.path_list_widget.mapToGlobal(position))
        if action == remove_action:
            item = self.path_list_widget.currentItem()
            if item:
                self.path_list_widget.takeItem(self.path_list_widget.row(item))

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
            items = [self.path_list_widget.item(i).text() for i in range(self.path_list_widget.count())]
            if path not in items:
                self.path_list_widget.addItem(path)
    
    def _start_scanning(self):
        paths = [self.path_list_widget.item(i).text() for i in range(self.path_list_widget.count())]
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
        
        # Update button to "Cancel Scanning"
        self.scan_btn.setText("Cancel Scanning")
        self.scan_btn.setEnabled(True)
        # Disconnect previous connection and connect to cancel function
        self.scan_btn.clicked.disconnect()
        self.scan_btn.clicked.connect(self._cancel_scanning)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting scan...")
        
        use_imagehash = self.phash_check.isChecked()
        
        # Clear previous results
        self.file_groups = {}
        self.file_groups_raw = {}
        # Clear selection state when starting new scan
        self.selection_state.clear()
        # Reset delete button counter
        self._update_delete_ui()
        # Reset page navigation
        self.current_page = 0
        self.page_label.setText("1/1")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        # Hide image-specific buttons when starting new scan
        if hasattr(self, 'keep_best_btn'):
            self.keep_best_btn.setVisible(False)
        if hasattr(self, 'keep_raw_btn'):
            self.keep_raw_btn.setVisible(False)
        while self.results_layout.count() > 1:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Store scan thread reference for potential cancellation
        self._scan_thread = None
        
        # Run scan in background thread
        def scan_thread():
            try:
                print(f"DEBUG: Starting scan thread for paths: {valid_paths}")
                print(f"DEBUG: use_imagehash={use_imagehash}")
                # Ensure callbacks are set
                print(f"DEBUG: status_callback={self.engine.status_callback}")
                print(f"DEBUG: progress_callback={self.engine.progress_callback}")
                print(f"DEBUG: results_callback={self.engine.results_callback}")
                
                # Test callback immediately
                self.engine.status_callback("Scan thread started...")
                self.engine.progress_callback(0.01)
                
                self.engine.scan_duplicate_files(valid_paths, use_imagehash)
                print("DEBUG: Scan completed")
            except Exception as e:
                import traceback
                print(f"DEBUG: Scan thread exception: {e}")
                traceback.print_exc()
                # Report error via status callback
                self.engine.status_callback(f"Scan error: {e}")
                self.engine.results_callback({})
        
        self._scan_thread = threading.Thread(target=scan_thread, daemon=True)
        self._scan_thread.start()
        print(f"DEBUG: Scan thread started, thread={self._scan_thread}")
    
    def _on_progress(self, value: float):
        """Progress callback from engine (thread-safe via Signal)."""
        print(f"DEBUG: _on_progress called with value={value}")
        # Emit signal for thread-safe UI update
        self.progress_updated.emit(value)
    
    def _on_status(self, message: str):
        """Status callback from engine (thread-safe via Signal)."""
        print(f"DEBUG: _on_status called with message={message}")
        # Emit signal for thread-safe UI update
        self.status_updated.emit(str(message))
    
    def _on_results(self, duplicate_groups: Dict[str, List[str]]):
        """Results callback from engine (thread-safe via Signal)."""
        print(f"DEBUG: _on_results called with {len(duplicate_groups)} groups")
        # Emit signal for thread-safe UI update
        self.results_ready.emit(duplicate_groups)
    
    @Slot(float)
    def _on_progress_slot(self, value: float):
        """Slot for progress updates (runs on main thread)."""
        print(f"DEBUG: _on_progress_slot called with value={value}")
        self.progress_bar.setValue(int(value * 100))
    
    @Slot(str)
    def _on_status_slot(self, message: str):
        """Slot for status updates (runs on main thread)."""
        print(f"DEBUG: _on_status_slot called with message={message}")
        self.status_label.setText(message)
    
    @Slot(dict)
    def _on_results_slot(self, duplicate_groups: Dict[str, List[str]]):
        """Slot for results updates (runs on main thread)."""
        print(f"DEBUG: _on_results_slot called with {len(duplicate_groups)} groups")
        self._display_results(duplicate_groups)
    
    
    def _cancel_scanning(self):
        """Cancel the current scanning process."""
        print("DEBUG: Cancel scanning requested")
        # Set cancellation flag in engine
        self.engine.scan_cancelled = True
        
        # Update UI
        self.scan_btn.setText("Start Scanning")
        self.scan_btn.setEnabled(True)
        # Reconnect button to start scanning
        self.scan_btn.clicked.disconnect()
        self.scan_btn.clicked.connect(self._start_scanning)
        
        self.progress_bar.setVisible(False)
        self.status_label.setText("Scan cancelled")
        
        # Clear scan thread reference
        self._scan_thread = None
    
    def _display_results(self, duplicate_groups: Dict[str, List[str]]):
        """Display scan results."""
        print(f"DEBUG: _display_results called with {len(duplicate_groups)} groups")
        
        # Restore button to "Start Scanning" state
        self.scan_btn.setText("Start Scanning")
        self.scan_btn.setEnabled(True)
        # Reconnect button to start scanning
        self.scan_btn.clicked.disconnect()
        self.scan_btn.clicked.connect(self._start_scanning)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        self.file_groups_raw = duplicate_groups
        
        # Apply current sorting
        current_sort = self.sort_combo.currentText()
        print(f"DEBUG: Applying current sorting: {current_sort}")
        self.file_groups = self.engine.apply_sorting(
            duplicate_groups,
            current_sort,
            group_by_type=False
        )
        
        print(f"DEBUG: Enabling scan button, hiding progress bar")
        self.progress_bar.setValue(0)
        
        if not duplicate_groups:
            print(f"DEBUG: No duplicate groups, setting status")
            self.status_label.setText("No duplicate files found.")
            # Hide image-specific buttons when no results
            self.keep_best_btn.setVisible(False)
            self.keep_raw_btn.setVisible(False)
            return
        
        print(f"DEBUG: Setting status and rendering page")
        self.status_label.setText(f"Found {len(duplicate_groups)} duplicate groups")
        
        # Update button visibility based on applicable files
        self._update_image_button_visibility()
        
        try:
            self._render_page(0)
        except Exception as e:
            print(f"DEBUG: Error in _render_page: {e}")
            import traceback
            traceback.print_exc()
        print(f"DEBUG: _display_results completed")
    
    def _render_page(self, page_index: int):
        """Render a page of groups."""
        print(f"DEBUG: _render_page called with page_index={page_index}, file_groups={len(self.file_groups) if self.file_groups else 0}")
        # Clear existing
        self._group_widgets.clear()
        print(f"DEBUG: Clearing {self.results_layout.count() - 1} existing widgets")
        while self.results_layout.count() > 1:  # Keep stretch
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self.file_groups:
            print(f"DEBUG: No file_groups to render")
            return
        
        all_groups = list(self.file_groups.items())
        total = len(all_groups)
        print(f"DEBUG: Total groups: {total}")
        max_page = (total - 1) // self.groups_per_page if total else 0
        page_index = max(0, min(page_index, max_page))
        self.current_page = page_index
        
        start = page_index * self.groups_per_page
        end = min(total, start + self.groups_per_page)
        page_groups = all_groups[start:end]
        print(f"DEBUG: Rendering groups {start} to {end} ({len(page_groups)} groups)")
        
        # Update pager
        total_pages = max_page + 1 if total > 0 else 1
        self.page_label.setText(f"{page_index + 1}/{total_pages}")
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
            for file_path in files[:12]:  # Limit preview
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
                for file_path in files[:12]:  # Limit preview
                    self._request_thumbnail(file_path, group_widget)
            except Exception as e:
                print(f"DEBUG: Error creating GroupWidget for {group_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Reset scroll to top
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(0))
        
        # Set initial status - show thumbnail loading if applicable, otherwise show groups info
        if self._thumbnail_total > 0:
            status_text = f"Loading thumbnails: 0/{self._thumbnail_total}"
        else:
            status_text = f"Groups {start+1}-{end} / {total}  (Page {page_index+1}/{total_pages})"
        print(f"DEBUG: Setting status to: {status_text}")
        self.status_label.setText(status_text)
        print(f"DEBUG: _render_page completed, created {len(self._group_widgets)} group widgets")
        
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
                self.status_label.setText(f"Loading thumbnails: {started_count}/{self._thumbnail_total}")
        
        # Submit worker
        worker = ThumbnailWorker(file_path, 300, self._thumb_page_token)
        
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
        print(f"DEBUG: _on_thumb_ready called for {os.path.basename(path)}, aspect={aspect:.2f}, pixmap size={pixmap.size().width()}x{pixmap.size().height()}, group_widget={group_widget is not None}")
        
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
                    self.status_label.setText(f"Loading thumbnails: {self._thumbnail_loaded}/{self._thumbnail_total}")
                else:
                    # All thumbnails loaded, restore normal status
                    if self.file_groups:
                        all_groups = list(self.file_groups.items())
                        total = len(all_groups)
                        start = self.current_page * self.groups_per_page
                        end = min(total, start + self.groups_per_page)
                        total_pages = max(0, (total - 1) // self.groups_per_page) + 1 if total > 0 else 1
                        self.status_label.setText(f"Groups {start+1}-{end} / {total}  (Page {self.current_page+1}/{total_pages})")
        
        # Check if stale - if group_widget is provided and still valid, always accept
        # Otherwise, check time difference (allow up to 60 seconds for slow loading)
        if group_widget is None:
            current_time = time.monotonic()
            time_diff = abs(current_time - self._thumb_page_token)
            if time_diff > 60.0:
                print(f"DEBUG: Thumbnail for {os.path.basename(path)} is stale (diff={time_diff:.2f}s), skipping")
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
                    print(f"DEBUG: Updated thumbnail for {os.path.basename(path)} in provided group widget")
                except RuntimeError:
                    # Widget was deleted, ignore
                    pass
        else:
            # Find widget containing this file
            for widget in list(self._group_widgets.values()):  # Create a copy to avoid modification during iteration
                if path in widget.files:
                    try:
                        widget.set_thumbnail(path, pixmap, aspect)
                        print(f"DEBUG: Updated thumbnail for {os.path.basename(path)} in found group widget")
                        break
                    except RuntimeError:
                        # Widget was deleted, continue to next
                        continue
    
    def _on_meta_ready(self, path: str, meta: str, group_widget: Optional[GroupWidget] = None):
        """Handle metadata ready signal."""
        print(f"DEBUG: _on_meta_ready called for {os.path.basename(path)}, meta={meta}, group_widget={group_widget is not None}")
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
        if event.button() == Qt.MouseButton.LeftButton:
            edge = self._get_resize_edge(event.pos())
            if edge:
                self._resize_edge_active = edge
                self._resize_start_pos = event.globalPos()
                self._resize_start_geom = self.geometry()
                event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
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
        margin = 8
        w, h = self.width(), self.height()
        x, y = pos.x(), pos.y()
        
        edge = []
        if x < margin: edge.append('left')
        elif x > w - margin: edge.append('right')
        if y < margin: edge.append('top')
        elif y > h - margin: edge.append('bottom')
        
        return '_'.join(edge) if edge else None

    def _update_delete_ui(self):
        """Update the delete and clear selection buttons based on state."""
        selected_count = sum(1 for v in self.selection_state.values() if v)
        self.delete_btn.setEnabled(selected_count > 0)
        if selected_count > 0:
            self.delete_btn.setText(f"Delete ({selected_count})")
        else:
            self.delete_btn.setText("Delete")
        
        # Also update clear selection button
        if hasattr(self, 'clear_select_btn'):
            self.clear_select_btn.setEnabled(selected_count > 0)
    
    def _update_image_button_visibility(self):
        """Update visibility of Keep Best and Keep RAW buttons based on applicable files."""
        if not self.file_groups:
            self.keep_best_btn.setVisible(False)
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
        self.keep_raw_btn.setVisible(has_raw_mixed)

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
                for path in selected:
                    try:
                        send2trash.send2trash(path)
                    except Exception as e:
                        print(f"Failed to delete {path}: {e}")
                dialog = CustomDialog(self, title="Done", message=f"Deleted {len(selected)} files", buttons=["OK"])
                dialog.exec()
                # Refresh
                self._refresh_display()
            except ImportError:
                dialog = CustomDialog(self, title="Error", message="send2trash not available", buttons=["OK"])
                dialog.exec()


def main():
    """Main entry point."""
    if not PYSIDE6_AVAILABLE:
        print("PySide6 not available. Please install: pip install PySide6")
        return 1
    
    print("Starting CloneWiper (Material 3)...")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look
    
    # Material Design 3 global styles
    app.setStyleSheet(f"""
        QWidget {{
            font-family: 'Roboto', 'Segoe UI', -apple-system, sans-serif;
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



