@echo off
REM Optimized build script for Windows EXE using PyInstaller
REM This version uses a spec file for better control over what gets included

echo ========================================
echo CloneWiper Windows Build Script (Optimized)
echo ========================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller. Please install manually: pip install pyinstaller
        pause
        exit /b 1
    )
)

REM Check if favicon.ico exists
set ICON_ARG=
set ICON_DATA=
if exist "icons\favicon.ico" (
    set ICON_ARG=--icon=icons\favicon.ico
    set ICON_DATA=icons\favicon.ico
) else if exist "favicon.ico" (
    set ICON_ARG=--icon=favicon.ico
    set ICON_DATA=favicon.ico
) else (
    echo Warning: favicon.ico not found. Building without icon...
)

echo.
echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "CloneWiper.spec" del /q "CloneWiper.spec"

echo.
echo Building executable (optimized for size)...
echo This may take several minutes...
echo.

pyinstaller --onefile ^
    --windowed ^
    --name=CloneWiper ^
    %ICON_ARG% ^
    --add-data "core;core" ^
    --add-data "%ICON_DATA%;." ^
    --hidden-import=PySide6.QtCore ^
    --hidden-import=PySide6.QtGui ^
    --hidden-import=PySide6.QtWidgets ^
    --hidden-import=PIL ^
    --hidden-import=PIL.Image ^
    --hidden-import=imagehash ^
    --hidden-import=rawpy ^
    --hidden-import=send2trash ^
    --hidden-import=mutagen ^
    --hidden-import=mutagen.mp3 ^
    --hidden-import=mutagen.flac ^
    --hidden-import=mutagen.mp4 ^
    --hidden-import=mutagen.oggvorbis ^
    --hidden-import=mutagen.oggopus ^
    --hidden-import=fitz ^
    --hidden-import=PyMuPDF ^
    --hidden-import=pypdfium2 ^
    --collect-binaries=PySide6 ^
    --collect-data=PySide6 ^
    --collect-data=PIL ^
    --collect-all=PyMuPDF ^
    --collect-all=mutagen ^
    --exclude-module=PySide6.scripts.deploy_lib ^
    --exclude-module=PySide6.QtBluetooth ^
    --exclude-module=PySide6.QtDBus ^
    --exclude-module=PySide6.QtDesigner ^
    --exclude-module=PySide6.QtHelp ^
    --exclude-module=PySide6.QtLocation ^
    --exclude-module=PySide6.QtMultimedia ^
    --exclude-module=PySide6.QtMultimediaWidgets ^
    --exclude-module=PySide6.QtNfc ^
    --exclude-module=PySide6.QtOpenGL ^
    --exclude-module=PySide6.QtPositioning ^
    --exclude-module=PySide6.QtQml ^
    --exclude-module=PySide6.QtQuick ^
    --exclude-module=PySide6.QtQuickWidgets ^
    --exclude-module=PySide6.QtRemoteObjects ^
    --exclude-module=PySide6.QtSensors ^
    --exclude-module=PySide6.QtSerialPort ^
    --exclude-module=PySide6.QtSql ^
    --exclude-module=PySide6.QtSvg ^
    --exclude-module=PySide6.QtTest ^
    --exclude-module=PySide6.QtWebChannel ^
    --exclude-module=PySide6.QtWebEngine ^
    --exclude-module=PySide6.QtWebEngineCore ^
    --exclude-module=PySide6.QtWebEngineWidgets ^
    --exclude-module=PySide6.QtWebSockets ^
    --exclude-module=PySide6.QtXml ^
    --exclude-module=PySide6.QtXmlPatterns ^
    --exclude-module=torch ^
    --exclude-module=torchvision ^
    --exclude-module=torchaudio ^
    --exclude-module=tensorboard ^
    --exclude-module=matplotlib ^
    --exclude-module=jupyter ^
    --exclude-module=notebook ^
    --exclude-module=IPython ^
    --exclude-module=opencv ^
    --exclude-module=cv2 ^
    --exclude-module=numpy ^
    --exclude-module=scipy ^
    --exclude-module=pandas ^
    --exclude-module=sklearn ^
    --exclude-module=skimage ^
    --exclude-module=tkinter ^
    --exclude-module=unittest ^
    --exclude-module=test ^
    --exclude-module=distutils ^
    --exclude-module=setuptools ^
    --exclude-module=pkg_resources ^
    --exclude-module=email ^
    --exclude-module=http ^
    --exclude-module=urllib3 ^
    --exclude-module=requests ^
    --noconfirm ^
    main.py

if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executable location: dist\CloneWiper.exe
echo.
echo Cleaning build directory (not needed after build)...
if exist "build" rmdir /s /q "build"
if exist "CloneWiper.spec" del /q "CloneWiper.spec"
echo.
echo Note: If the file is still large, consider:
echo   - Using UPX compression (if available)
echo   - Creating a spec file for more fine-grained control
echo   - Removing unused dependencies from your environment
echo.
pause

