# CloneWiper 🧹

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows-blue?logo=windows)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-orange?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/markyip)
![Downloads](https://img.shields.io/github/downloads/markyip/CloneWiper/total) 

CloneWiper is a high-performance, modern duplicate file detection tool built with Python and PySide6 (Qt). It follows **Material Design 3** principles to provide a premium, seamless experience for managing your file library.

## ✨ Features

### Core Functionality
- **Smart Duplicate Detection**: Fast MD5 and multi-algorithm perceptual hashing for highly accurate duplicate detection
  - **Multi-Algorithm Perceptual Hashing**: Combines average_hash, phash (perceptual), dhash (difference), and whash (wavelet) for superior accuracy
  - **Image Support**: Works with common formats (JPEG, PNG, GIF, BMP, TIFF, WebP) and RAW files (CR2, NEF, ARW, etc.)
  - **Video Support**: Perceptual hashing for video files using keyframe extraction
- **Cross-Platform Support**: Works on Windows (macOS support from source code only)
- **High Performance**: Asynchronous processing with multi-threaded file scanning
- **Persistent Caching**: SQLite-backed cache for fast re-scans

### User Interface
- **Material Design 3 UI**: Clean, modern dark-themed interface
- **Custom Title Bar**: Frameless window with custom controls
- **Smart Thumbnails**: 
  - **Images**: Fast previews, including RAW support (`.arw`, `.cr2`, `.nef`, etc.)
  - **Video**: Frame extraction for common video formats
  - **Documents**: High-quality **PDF**, **EPUB**, **MOBI**, and **AZW3** thumbnails using **pypdfium2** and **PyMuPDF**
  - **Music**: Album art extraction and rich metadata display using **mutagen**
- **Interactive File Cards**: Hover effects, scrolling text for long filenames, and selection management
- **Pagination**: Efficient handling of large result sets
- **Quick Actions**: Keep Newest, Keep Oldest, Keep Best, Keep RAW, Delete Selected

### Advanced Features
- **Multi-Algorithm Perceptual Hashing**: Combines multiple hash algorithms (average, perceptual, difference, wavelet) to detect similar images and videos even if they're slightly modified, resized, or have different compression
- **File Type Grouping**: Organize duplicates by file type
- **Multiple Sorting Options**: Sort by count, size, name, or date
- **Safe Deletion**: Uses `send2trash` to move files to recycle bin/trash

## 📋 Prerequisites

- **Python 3.8+**
- **Windows 10/11** (macOS: run from source code only, executable build not currently supported)

## 🚀 Installation

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CloneWiper.git
   cd CloneWiper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Optional Dependencies (Recommended)

For full feature support, install optional dependencies:

```bash
# Video thumbnails
pip install opencv-python>=4.8.0

# PDF/EPUB/MOBI/AZW3 thumbnails
pip install PyMuPDF>=1.23.0
pip install pypdfium2>=0.20.0

# Music metadata and album art
pip install mutagen>=1.47.0
```

## 💻 Usage

### Windows
```bash
# Using launch script
launch.bat

# Or directly
python main.py
```

### macOS (Source Code Only)
**Note**: macOS executable build is currently not supported. You can run from source code:

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run directly
python3 main.py
```

## 🔨 Building Executables

### Windows (EXE)

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

2. Run the build script:
   ```bash
   build_windows.bat
   ```
   
   This build script will:
   - Check and install PyInstaller if needed
   - Build an optimized executable with all features
   - Exclude unnecessary modules to minimize file size
   
   **Note**: If your executable is larger than expected (>300MB), consider creating a clean virtual environment with only required dependencies before building.

   Or manually:
   ```bash
   pyinstaller --onefile --windowed --icon=favicon.ico --name=CloneWiper main.py
   ```

   The executable will be in `dist/CloneWiper.exe`

### macOS Build

**Note**: macOS executable build is currently not supported. Please run from source code using `python3 main.py`.

## 📁 Project Structure

```
CloneWiper/
├── core/
│   ├── __init__.py
│   └── engine.py          # Core scanning and hashing engine
├── icons/
│   └── README.md          # Icon resources documentation
├── main.py                 # Application entry point
├── qt_app.py              # PySide6 UI implementation
├── requirements.txt       # Python dependencies
├── favicon.ico           # Application icon (Windows)
├── launch.bat            # Windows launch script
├── build_windows.bat     # Windows build script
├── BUILD.md              # Build instructions
├── README.md             # This file
└── LICENSE               # License file
```

##  Development

### Running Tests
```bash
# Add tests when available
python -m pytest
```

### Code Style
This project follows PEP 8 style guidelines.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **PySide6** - Qt for Python
- **Pillow** - Image processing
- **ImageHash** - Perceptual hashing
- **PyMuPDF** - PDF/EPUB rendering
- **pypdfium2** - High-quality PDF rendering
- **Material Design 3** - Design guidelines

## 📧 Contact

For issues, questions, or suggestions, please open an issue on GitHub.

---
