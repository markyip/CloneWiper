# Icons Directory

This directory contains application icons for different platforms.

## Icon Files

- **favicon.ico**: Windows icon (also used as fallback)
- **app.icns**: macOS icon (recommended for better quality)

## Creating macOS Icon (.icns)

### Option 1: Using iconutil (macOS only)

1. Create an iconset directory:
   ```bash
   mkdir -p app.iconset
   ```

2. Add PNG images in different sizes:
   - `icon_16x16.png`
   - `icon_16x16@2x.png` (32x32)
   - `icon_32x32.png`
   - `icon_32x32@2x.png` (64x64)
   - `icon_128x128.png`
   - `icon_128x128@2x.png` (256x256)
   - `icon_256x256.png`
   - `icon_256x256@2x.png` (512x512)
   - `icon_512x512.png`
   - `icon_512x512@2x.png` (1024x1024)

3. Convert to .icns:
   ```bash
   iconutil --convert icns app.iconset --output app.icns
   ```

### Option 2: Using Online Tools

- Use online converters like:
  - https://cloudconvert.com/ico-to-icns
  - https://convertio.co/ico-icns/

### Option 3: Using Python (Pillow)

If you have a high-resolution PNG (1024x1024), you can use a script to generate all sizes and convert to .icns.

## Notes

- The build script will automatically use `app.icns` if available, otherwise fall back to `favicon.ico`
- For best results on macOS, use `.icns` format
- Windows builds use `favicon.ico` from the project root








