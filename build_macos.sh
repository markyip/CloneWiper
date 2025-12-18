#!/bin/bash

# Build script for macOS DMG using PyInstaller

set -e  # Exit on error

echo "========================================"
echo "CloneWiper macOS Build Script"
echo "========================================"
echo ""

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller not found. Installing..."
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo "Failed to install PyInstaller. Please install manually: pip3 install pyinstaller"
        exit 1
    fi
fi

# Check if create-dmg is installed
if ! command -v create-dmg &> /dev/null; then
    echo "create-dmg not found. Installing..."
    if command -v brew &> /dev/null; then
        brew install create-dmg
    else
        echo "Please install create-dmg manually:"
        echo "  npm install -g create-dmg"
        echo "  or"
        echo "  brew install create-dmg"
        exit 1
    fi
fi

# Check for icon files (prefer .icns for macOS, fallback to .ico)
ICON_ARG=""
ICON_DATA=""
if [ -f "icons/app.icns" ]; then
    echo "Using icons/app.icns for macOS build..."
    ICON_ARG="--icon=icons/app.icns"
    ICON_DATA="icons/app.icns"
elif [ -f "app.icns" ]; then
    echo "Using app.icns for macOS build..."
    ICON_ARG="--icon=app.icns"
    ICON_DATA="app.icns"
elif [ -f "icons/favicon.ico" ]; then
    echo "Using icons/favicon.ico for macOS build (consider creating app.icns for better quality)..."
    ICON_ARG="--icon=icons/favicon.ico"
    ICON_DATA="icons/favicon.ico"
elif [ -f "favicon.ico" ]; then
    echo "Using favicon.ico for macOS build (consider creating app.icns for better quality)..."
    ICON_ARG="--icon=favicon.ico"
    ICON_DATA="favicon.ico"
else
    echo "Warning: No icon file found. Building without icon..."
    echo "  Recommended: Create icons/app.icns for best macOS appearance"
fi

echo ""
echo "Cleaning previous builds..."
rm -rf build dist CloneWiper.spec CloneWiper.app

echo ""
echo "Building application bundle..."

# Build PyInstaller command
# Use --onedir (not --onefile) for macOS to create proper .app bundle
# Add --osx-bundle-identifier for proper macOS app bundle
PYINSTALLER_CMD="pyinstaller --onedir --windowed --name=CloneWiper --osx-bundle-identifier=com.clonewiper.app"

# Add icon if specified
if [ -n "$ICON_ARG" ]; then
    PYINSTALLER_CMD="$PYINSTALLER_CMD $ICON_ARG"
fi

# Add data files
PYINSTALLER_CMD="$PYINSTALLER_CMD --add-data \"core:core\""
if [ -n "$ICON_DATA" ]; then
    PYINSTALLER_CMD="$PYINSTALLER_CMD --add-data \"$ICON_DATA:.\""
fi

# Add .icns file if it exists
if [ -f "icons/app.icns" ]; then
    PYINSTALLER_CMD="$PYINSTALLER_CMD --add-data \"icons/app.icns:.\""
elif [ -f "app.icns" ]; then
    PYINSTALLER_CMD="$PYINSTALLER_CMD --add-data \"app.icns:.\""
fi

# Add hidden imports and other options
PYINSTALLER_CMD="$PYINSTALLER_CMD --hidden-import=PySide6.QtCore \
    --hidden-import=PySide6.QtGui \
    --hidden-import=PySide6.QtWidgets \
    --hidden-import=PIL \
    --hidden-import=imagehash \
    --hidden-import=rawpy \
    --hidden-import=send2trash \
    --collect-all=PySide6 \
    --collect-all=PIL \
    --exclude-module=PySide6.scripts.deploy_lib \
    --exclude-module=torch \
    --exclude-module=torchvision \
    --exclude-module=torchaudio \
    --exclude-module=tensorboard \
    --exclude-module=matplotlib \
    --exclude-module=jupyter \
    --exclude-module=notebook \
    --exclude-module=IPython \
    --noconfirm \
    main.py"

# Execute the command
eval $PYINSTALLER_CMD

if [ $? -ne 0 ]; then
    echo ""
    echo "Build failed!"
    exit 1
fi

# PyInstaller creates a .app bundle on macOS
if [ -d "dist/CloneWiper.app" ]; then
    echo ""
    echo "Post-processing app bundle..."
    
    # Fix permissions for the executable
    if [ -f "dist/CloneWiper.app/Contents/MacOS/CloneWiper" ]; then
        chmod +x "dist/CloneWiper.app/Contents/MacOS/CloneWiper"
        echo "Set executable permissions on CloneWiper binary"
    fi
    
    # Create Info.plist if it doesn't exist or update it
    INFO_PLIST="dist/CloneWiper.app/Contents/Info.plist"
    if [ -f "$INFO_PLIST" ]; then
        # Update minimum macOS version if needed
        /usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion 10.14" "$INFO_PLIST" 2>/dev/null || \
        /usr/libexec/PlistBuddy -c "Add :LSMinimumSystemVersion string 10.14" "$INFO_PLIST" 2>/dev/null
        echo "Updated Info.plist"
    fi
    
    # Remove quarantine attribute (allows app to run without Gatekeeper blocking)
    xattr -cr "dist/CloneWiper.app" 2>/dev/null || echo "Note: Could not remove quarantine attributes (may need to allow in System Settings)"
    
    echo ""
    echo "Creating DMG..."
    
    # Create DMG using create-dmg
    create-dmg \
        --volname "CloneWiper" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "CloneWiper.app" 175 190 \
        --hide-extension "CloneWiper.app" \
        --app-drop-link 425 190 \
        "dist/CloneWiper.dmg" \
        "dist/CloneWiper.app" 2>&1
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Cleaning build directory (not needed after build)..."
        rm -rf build CloneWiper.spec
        echo ""
        echo "========================================"
        echo "Build completed successfully!"
        echo "========================================"
        echo ""
        echo "DMG location: dist/CloneWiper.dmg"
        echo "App bundle: dist/CloneWiper.app"
        echo ""
        echo "IMPORTANT: If the app doesn't launch when double-clicked:"
        echo "1. Right-click the app and select 'Open' (first time only)"
        echo "2. Or go to System Settings > Privacy & Security and allow the app"
        echo "3. Or run from terminal: open dist/CloneWiper.app"
        echo ""
    else
        echo ""
        echo "DMG creation failed, but app bundle is available at: dist/CloneWiper.app"
        echo ""
        echo "To test the app bundle:"
        echo "  open dist/CloneWiper.app"
        echo "  or"
        echo "  dist/CloneWiper.app/Contents/MacOS/CloneWiper"
        echo ""
    fi
else
    echo ""
    echo "App bundle not found. Build may have failed."
    echo "Check the error messages above for details."
    exit 1
fi

