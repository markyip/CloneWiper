# CloneWiper
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows-blue?logo=windows)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-orange?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/markyip)
![Downloads](https://img.shields.io/github/downloads/markyip/CloneWiper/total) 

CloneWiper is a high-performance, modern duplicate file detection tool built with Python and PySide6 (Qt). It follows **Material Design 3** principles to provide a premium, seamless experience for managing your file library.

## ✨ Features

### Core Functionality
- **Smart Duplicate Detection**: Three hash modes for flexible duplicate detection
  - **MD5 Only**: Fast exact duplicate detection using MD5 checksums (best for identical files)
  - **Single Perceptual Hash**: Detects visually similar images using average hash algorithm
  - **Multi-Algorithm Perceptual Hashing** (Default): Combines four algorithms (average_hash, phash, dhash, whash) with voting mechanism for superior accuracy
    - Uses Hamming distance comparison with voting (requires 3/4 algorithms to agree)
    - Detects duplicates even when images are resized, compressed, or slightly modified
    - Optimized with parallel hash calculation and two-phase filtering
  - **Image Support**: Works with common formats (JPEG, PNG, GIF, BMP, TIFF, WebP) and RAW files (CR2, NEF, ARW, etc.)
  - **Video Support**: Perceptual hashing for video files using keyframe extraction
- **Cross-Platform Support**: Works on Windows (macOS support from source code only)
- **High Performance**: 
  - Asynchronous processing with multi-threaded file scanning
  - Dynamic CPU optimization for hybrid architectures (P-cores/E-cores detection)
  - Adaptive I/O strategy (preloads small files, chunks large files)
  - Batch cache writes to reduce database lock contention
- **Persistent Caching**: SQLite-backed cache for fast re-scans (hashes are cached and persist across sessions)

### User Interface
- **Material Design 3 UI**: Clean, modern dark-themed interface with rounded corners (when not maximized)
- **Custom Title Bar**: Frameless window with custom controls and window management
- **Smart Thumbnails**: 
  - **Images**: Fast previews, including RAW support (`.arw`, `.cr2`, `.nef`, etc.)
  - **Video**: Frame extraction for common video formats
  - **Documents**: High-quality **PDF**, **EPUB**, **MOBI**, and **AZW3** thumbnails using **pypdfium2** and **PyMuPDF**
  - **Music**: Album art extraction and rich metadata display using **mutagen**
- **Interactive File Cards**: Hover effects, scrolling text for long filenames, and selection management
- **Pagination**: Efficient handling of large result sets with clickable page indicator dropdown
- **Drag & Drop**: Drag and drop folders onto the results area for easy folder selection
- **Real-Time Progress**: Centered progress indicator with adaptive update intervals
- **Quick Selection Strategies**:
  - **Keep Newest**: Keeps the most recently modified file
  - **Keep Oldest**: Keeps the oldest file by modification time
  - **Keep Best**: Keeps the highest resolution image; if multiple share the highest resolution, keeps the largest file size
  - **Keep Smallest**: Keeps the highest resolution image; if multiple share the highest resolution, keeps the smallest file size
  - **Keep RAW**: Prefers RAW files over JPEG when both exist in the same group
- **Quick Actions**: Delete Selected, Clear Selection (with scope: Current Page or All Pages)

### Advanced Features
- **Multi-Algorithm Perceptual Hashing**: 
  - Combines four hash algorithms (average, perceptual, difference, wavelet) with parallel calculation
  - Uses Hamming distance comparison with voting mechanism (requires 3/4 algorithms to agree)
  - Two-phase filtering: quick filter with average_hash, then detailed multi-algorithm comparison
  - Detects similar images and videos even if they're slightly modified, resized, or have different compression
- **Hybrid CPU Optimization**: Automatically detects and optimizes for hybrid CPU architectures (Intel 12th/13th gen, AMD Ryzen)
  - Dynamically adjusts worker threads based on P-cores and E-cores
  - Optimized thread pool sizes for I/O-intensive and CPU-intensive tasks
- **File Type Grouping**: Organize duplicates by file type
- **Multiple Sorting Options**: Sort by count, size, name, or date (ascending/descending)
- **Scope Control**: Apply actions to current page or all pages
- **Safe Deletion**: Uses `send2trash` to move files to recycle bin/trash
- **Persistent Cache**: 
  - SQLite database stores calculated hashes (p-hash and MD5)
  - Cache persists across sessions - no need to recalculate hashes on re-scan
  - Automatic cache management with hit/miss statistics

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

### Hash Mode Selection

CloneWiper offers three hash modes for different use cases:

1. **MD5 Only** (Fastest)
   - Best for: Finding exact duplicate files
   - Uses: MD5 checksum comparison
   - Pros: Very fast, low CPU usage
   - Cons: Only detects identical files (byte-for-byte)

2. **Single Perceptual Hash** (Balanced)
   - Best for: Finding visually similar images with moderate accuracy
   - Uses: Average hash algorithm
   - Pros: Faster than multi-algorithm, detects resized/compressed images
   - Cons: Less accurate than multi-algorithm mode

3. **Multi-Algorithm Perceptual Hash** (Most Accurate - Default)
   - Best for: Finding visually similar images with highest accuracy
   - Uses: Four algorithms (average, perceptual, difference, wavelet) with voting
   - Pros: Highest accuracy, detects duplicates even with modifications
   - Cons: Slower than other modes (but optimized with parallel processing)

**Recommendation**: Use Multi-Algorithm Perceptual Hash for most cases, as it provides the best balance of accuracy and performance with caching enabled.

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
- **rawpy** - RAW image processing
- **OpenCV** - Video processing
- **mutagen** - Audio metadata
- **psutil** - CPU architecture detection
- **Material Design 3** - Design guidelines

## 📧 Contact

For issues, questions, or suggestions, please open an issue on GitHub.

---
