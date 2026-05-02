# CloneWiper v1.2 - Release notes

**Release date:** May 2, 2026

## Highlights

- **Native Windows window behavior** for the frameless Qt window: edge resize, drag-to-snap, and Windows 11 Snap Layout support.
- **More polished Material Design 3 UI** with corrected rounded corners, transparent dialogs, clearer quick-select scope, and highlighted quick-select actions.
- **Faster thumbnail handling** for expensive media previews while keeping regular image thumbnails memory-only.
- **Faster delete flow** using batch recycle-bin operations and cache cleanup for deleted files.
- **Larger result pages**: pagination now shows 100 duplicate groups per page.

---

## UI and layout

- Fixed frameless window hit-testing on Windows so users can resize by dragging edges and use OS snap features.
- Fixed in-app dialog corners by rendering dialogs as transparent frameless windows with an inner rounded card.
- Aligned thumbnail-card radii so nested rounded corners follow the actual card padding.
- Fixed group headers so their background no longer blocks the parent card's rounded corners.
- Fixed collapsed groups so they can expand again after the list view height is restored and recalculated.
- Clarified quick-select scope with a `Current Page` / `All Pages` label.
- Highlighted the most recently applied `Keep ...` quick-select strategy.
- Added inline `x` controls for scan folders, plus Backspace support for removing the selected folder.

---

## Thumbnail performance and cache policy

- Persistent thumbnail cache now stores only expensive previews:
  - PDF / EPUB / MOBI / AZW3
  - Video thumbnails
  - Audio album artwork
- Regular image and RAW thumbnails are memory-only to avoid growing persistent storage for files users are likely to delete.
- Deleted files are removed from memory thumbnail caches and the persistent thumbnail cache.
- Startup cache maintenance prunes unsupported thumbnail formats, removes entries not accessed for 30 days, and vacuums the database to reclaim space.
- Scan-completion cache maintenance performs lightweight pruning without vacuuming, avoiding unnecessary UI pauses.
- PDF thumbnail rendering now scales closer to the target preview size instead of always rendering at a fixed high scale.
- Video thumbnail extraction now opens each video once for both metadata and the preview frame.
- SQLite thumbnail-cache hits update `last_access` lazily to reduce write amplification during large gallery loads.

---

## Delete flow

- Delete operations now try a batch `send2trash` call first, which is significantly faster on Windows for large selections.
- If batch delete fails, CloneWiper falls back to per-file delete to preserve error handling.
- Result groups are updated directly from the deleted-path set without extra `os.path.exists()` checks for every remaining file.
- Failed delete items remain selected so users can retry.

---

## Result browsing

- Result pages now show **100 groups per page** instead of 50.
- Memory thumbnail cache was increased to better cover the larger page size.

---

## Documentation

- README updated for v1.2, including the refined persistent thumbnail cache policy, Windows window behavior, folder removal controls, quick-select highlighting, and 100-group pagination.

---

Thank you for using CloneWiper. Issues and PRs welcome on [GitHub](https://github.com/markyip/CloneWiper).
