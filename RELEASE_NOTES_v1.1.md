# CloneWiper v1.1 — Release notes

**Release date:** April 9, 2026  

## Highlights

- **Faster scans and comparisons** in the core engine (partial hashing, similarity grouping, and parallel hash-pair processing).
- **Quieter default output**: debug-style messages use Python `logging` instead of unconditional `print`.
- **Persistent thumbnail cache** in SQLite for quicker UI scrolling on repeat runs.
- **Windows build fix** for PyInstaller 6 on Python 3.12+ (`distutils` exclude removed).

---

## Performance

- Partial-hash phase uses `concurrent.futures.wait(..., FIRST_COMPLETED)` instead of a tight `as_completed` timeout loop, reducing wasted CPU while waiting for workers.
- Multi-algorithm similarity: **pre-parse** perceptual hashes once per file hash string; reuse for LSH buckets, parallel comparisons, and the small-dataset path (fewer repeated `hex_to_hash` calls).
- Candidate **hash-pair comparison** uses `as_completed` instead of `executor.map`, avoiding head-of-line blocking when pair cost varies (e.g. ORB).
- Full-hash aggregation no longer takes a global lock on every file (single-threaded consumer from `as_completed`).
- Background **file prefetch** for hashing is **disabled by default** (`_file_prefetch_enabled = False`) to avoid extra threads and redundant I/O; the prefetch pool shutdown behavior was corrected when prefetch is re-enabled in code.

---

## Logging and diagnostics

- Replaced pervasive `DEBUG:` `print` calls with **`logger.debug`** in `core/engine.py` and `qt_app.py`; thumbnail cache errors use `logger.debug` in `core/thumbnail_cache.py`.
- **Default**: no verbose console spam (standard logging levels).
- **Optional**: set environment variable **`CLONEWIPER_DEBUG=1`** (or `true`, `yes`, `on`) before launch; the Qt entrypoint configures `logging` at DEBUG when this is set.

---

## Thumbnail cache

- New module **`core/thumbnail_cache.py`**: SQLite-backed storage for generated thumbnails (platform-appropriate path under `%LOCALAPPDATA%\CloneWiper` on Windows).
- Optional script **`verify_thumbnail_cache.py`** to inspect or validate cache behavior.

---

## Windows executable build

- **`build_windows.bat`**: removed **`--exclude-module=distutils`**, which triggered  
  `ValueError: Target module "distutils" already imported as "ExcludedModule('distutils',)"`  
  under **PyInstaller 6** and **Python 3.12+** during dependency analysis.

---

## Documentation

- **README** updated: version **v1.1** badge, correct clone URL (`markyip/CloneWiper`), `CLONEWIPER_DEBUG` usage, build note for Python 3.12+, and project layout including thumbnail cache files.

---

## Upgrade notes

- No change required to existing **hash** SQLite caches; they continue to work as before.
- First run after upgrade may populate the **thumbnail** cache; subsequent sessions should feel snappier when browsing large result sets.

---

## Full list of touched areas (summary)

| Area | Change |
|------|--------|
| `core/engine.py` | Scan/compare performance, logging, prefetch default off |
| `qt_app.py` | Logging, `CLONEWIPER_DEBUG`, traceback via logger |
| `core/thumbnail_cache.py` | New persistent thumbnail cache |
| `verify_thumbnail_cache.py` | New optional utility |
| `build_windows.bat` | PyInstaller / `distutils` fix |
| `README.md` | v1.1, docs, structure |

---

Thank you for using CloneWiper. Issues and PRs welcome on [GitHub](https://github.com/markyip/CloneWiper).
