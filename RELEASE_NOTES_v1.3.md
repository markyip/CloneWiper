# CloneWiper v1.3 — Release notes

**Release date:** June 13, 2026

## Highlights

- **Much faster large-library scans** via tiered pHash screening, batched prefiltering, and smarter similarity grouping.
- **Reliable scan progress** on huge folders (fine-grained status updates through prefilter and pHash phases).
- **Fixed Keep Best / Keep Smallest** for identical-resolution duplicates, RAW files, and EXIF-oriented dimensions.
- **Windows executable icon** now appears correctly in the taskbar, title bar, and Alt+Tab.
- **ORB hash modes** exposed in the UI (single and multi-algorithm variants with ORB verification).

---

## Scan performance

- **Tier-1 pHash screening**: computes a fast perceptual hash first and skips full multi-hash work for images with a unique pHash.
- **Batched prefilter**: reuses Phase-1 size groups directly instead of re-indexing hundreds of thousands of files per size bucket.
- **Union-Find grouping**: builds similarity clusters from candidate pairs without arbitrary alphabetical truncation.
- **Bounded parallel pair comparison**: priority-ordered candidate pairs with inflight limits (up to 50,000 pairs) for steadier throughput on large sets.
- **Reduced redundant I/O**: partial-hash and ORB stages reuse cached file metadata where possible.
- **pHash cache migration**: safer SQLite schema upgrade from `PRIMARY KEY (path)` to `PRIMARY KEY (path, algo)` with cleanup of failed migration attempts.

---

## Scan progress reporting

- New phase helpers (`_begin_scan_phase`, `_report_scan_progress`) throttle UI updates while staying responsive on 100k+ file scans.
- Prefilter reports progress across multi-file size groups instead of appearing stuck at 0%.
- pHash pre-screen splits into three visible stages: hash calculation, similarity index build, and bucket comparison.
- Center status text now includes a **percentage** alongside the phase detail during scans.

---

## Quick selection (Keep Best / Keep Smallest / Keep RAW)

- **Consistent dimension reading**: RAW files use **rawpy** sensor dimensions (matching thumbnail metadata); standard images apply **EXIF orientation** before measuring width × height.
- **Exact dimension ties**: images with the same width and height are treated as the same resolution tier before file-size tie-breaking.
- **Full-group cleanup**: when an image is kept, all other files in the group (including sidecars such as `.xmp`) are marked for deletion.
- Thumbnail metadata for standard images also uses EXIF transpose so displayed dimensions match quick-select logic.

---

## UI and layout

- **Application icon**: centralized favicon resolution for dev and PyInstaller bundles; `QApplication.setWindowIcon()` plus Windows `AppUserModelID` for correct taskbar branding.
- **Footer toolbar**: horizontally scrollable on narrow windows; hidden during scans and shown when results are ready.
- **Group headers**: bottom corners round correctly when a group is collapsed.
- **Central widget styling**: background color scoped to `#centralWidget` so child cards keep their own surfaces.
- **Folder list**: paths stored in item `UserRole` to avoid duplicate text rendering with custom row widgets.
- Quick-select highlight resets when starting a new scan or changing pages in **Current Page** scope.
- Windows **DWM rounded corners** applied to the frameless window.

---

## Windows build

- `build_windows.bat` bundles `icons\favicon.ico` into the `icons/` folder inside the executable (fixes missing taskbar icon in packaged builds).
- `favicon.ico` regenerated with multiple embedded sizes (16–256 px).

---

## Thumbnail cache

- `VACUUM` now runs outside an active SQLite transaction during cache cleanup, preventing silent vacuum failures.

---

## Documentation

- README updated for v1.3: scan performance, progress reporting, five hash modes (including ORB), Keep Best behavior, Windows icon fix, and footer/scan UX changes.

---

## Upgrade notes

- No action required for existing hash caches; pHash schema migration runs automatically on first launch if needed.
- Rebuild Windows executables with the updated `build_windows.bat` to pick up the icon bundling fix.

---

## Files changed (summary)

| Area | Change |
|------|--------|
| `core/engine.py` | Tiered pHash, batched prefilter, Union-Find, progress reporting, Keep Best fixes |
| `qt_app.py` | Scan progress %, icon loading, EXIF metadata, footer/scan UX, ORB modes in UI |
| `core/thumbnail_cache.py` | VACUUM transaction fix |
| `build_windows.bat` | Correct favicon bundle path |
| `icons/favicon.ico` | Multi-size icon asset |
| `README.md` | v1.3 documentation |
| `RELEASE_NOTES_v1.3.md` | This file |

---

Thank you for using CloneWiper. Issues and PRs welcome on [GitHub](https://github.com/markyip/CloneWiper).
