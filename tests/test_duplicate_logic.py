"""Checks for duplicate grouping mistakes that used to drop or invent matches."""
import os
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from core.engine import ScanEngine


def _engine() -> ScanEngine:
    engine = ScanEngine()
    engine.phash_cache_enabled = False
    engine.md5_cache_enabled = False
    engine.opencv_verification_method = None
    return engine


def _pattern(size, background, boxes):
    image = Image.new("RGB", (size, size), background)
    pixels = image.load()
    for x0, y0, x1, y1, color in boxes:
        for y in range(int(y0 * size), int(y1 * size)):
            for x in range(int(x0 * size), int(x1 * size)):
                pixels[x, y] = color
    return image


def _sample_images():
    boxes = [
        (0.05, 0.1, 0.4, 0.9, (10, 10, 10)),
        (0.5, 0.05, 0.95, 0.45, (250, 250, 250)),
        (0.45, 0.55, 0.9, 0.95, (220, 30, 30)),
    ]
    original = _pattern(160, (70, 130, 190), boxes)
    near = original.resize((72, 72), Image.Resampling.BILINEAR)
    pixels = near.load()
    for index in range(8):
        pixels[index, 0] = (0, 0, 0)
        pixels[index, 1] = (255, 255, 255)
    other = _pattern(
        160,
        (240, 210, 40),
        [
            (0.0, 0.0, 0.25, 1.0, (0, 80, 0)),
            (0.6, 0.0, 1.0, 0.4, (80, 0, 80)),
            (0.3, 0.6, 0.8, 1.0, (255, 255, 255)),
        ],
    )
    return original, near, other


class DuplicateGroupingTests(unittest.TestCase):
    def test_multi_hash_prefix_still_groups_similar_images(self):
        engine = _engine()
        original, near, other = _sample_images()
        red_hash = engine._calculate_multi_hash(original)
        small_hash = engine._calculate_multi_hash(near)
        blue_hash = engine._calculate_multi_hash(other)
        self.assertTrue(red_hash and small_hash and blue_hash)
        self.assertNotEqual(red_hash, small_hash)

        groups = engine._group_by_similarity_multi_hash(
            {
                f"img_{red_hash}": ["red-large.jpg"],
                f"img_{small_hash}": ["red-small.jpg"],
                f"img_{blue_hash}": ["blue.jpg"],
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": ["notes.pdf", "notes-copy.pdf"],
            },
            skip_orb_verification=True,
        )
        flat = {path for files in groups.values() for path in files}
        self.assertIn("notes.pdf", flat)
        self.assertIn("notes-copy.pdf", flat)
        red_group = next(files for files in groups.values() if "red-large.jpg" in files)
        self.assertIn("red-small.jpg", red_group)
        self.assertNotIn("blue.jpg", red_group)

    def test_single_hash_uses_hamming_distance(self):
        engine = _engine()
        original, near, other = _sample_images()
        hashes = {
            "red-large.jpg": engine._calculate_single_hash(original),
            "red-small.jpg": engine._calculate_single_hash(near),
            "blue.jpg": engine._calculate_single_hash(other),
        }
        self.assertTrue(all(hashes.values()))
        self.assertNotEqual(hashes["red-large.jpg"], hashes["red-small.jpg"])
        grouped_input = {}
        for path, digest in hashes.items():
            grouped_input.setdefault(f"img_{digest}", []).append(path)

        groups = engine._group_by_similarity_single_hash(grouped_input)
        red_group = next(files for files in groups.values() if "red-large.jpg" in files)
        self.assertIn("red-small.jpg", red_group)
        self.assertNotIn("blue.jpg", red_group)

    def test_same_size_is_not_enough_for_non_images(self):
        engine = _engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "a" / "song.mp3"
            second = Path(temp_dir) / "b" / "other.mp3"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"A" * 1000)
            second.write_bytes(b"B" * 1000)
            metadata = {
                str(first): (1000, 10.0),
                str(second): (1000, 20.0),
            }
            exact, remaining = engine._prefilter_from_size_groups(
                {1000: [str(first), str(second)]},
                metadata,
            )
            self.assertEqual(exact, {})
            self.assertCountEqual(remaining, [str(first), str(second)])

    def test_same_name_and_time_still_requires_matching_bytes(self):
        engine = _engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "a" / "song.mp3"
            second = Path(temp_dir) / "b" / "song.mp3"
            third = Path(temp_dir) / "c" / "song.mp3"
            for path in (first, second, third):
                path.parent.mkdir()
            first.write_bytes(b"SAME" * 250)
            second.write_bytes(b"SAME" * 250)
            third.write_bytes(b"DIFF" * 250)
            files = [str(first), str(second), str(third)]
            metadata = {path: (1000, 50.0) for path in files}
            exact, remaining = engine._prefilter_from_size_groups({1000: files}, metadata)
            self.assertEqual(len(exact), 1)
            grouped = next(iter(exact.values()))
            self.assertCountEqual(grouped, [str(first), str(second)])
            self.assertEqual(remaining, [str(third)])

    def test_exact_metadata_groups_are_not_discarded(self):
        engine = _engine()
        found = {}

        def _capture(groups):
            found["groups"] = groups

        engine.results_callback = _capture
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "a" / "song.mp3"
            second = Path(temp_dir) / "b" / "song.mp3"
            first.parent.mkdir()
            second.parent.mkdir()
            payload = b"SAME" * 250
            first.write_bytes(payload)
            second.write_bytes(payload)
            stamp = time.time()
            os.utime(first, (stamp, stamp))
            os.utime(second, (stamp, stamp))
            engine.scan_duplicate_files([temp_dir], use_imagehash=False, use_multi_hash=False)

        groups = found["groups"]
        self.assertEqual(len(groups), 1)
        self.assertCountEqual(next(iter(groups.values())), [str(first), str(second)])

    def test_raw_jpeg_merge_keeps_camera_numbers(self):
        engine = _engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "IMG_0001.CR2"
            same_jpeg = root / "IMG_0001.JPG"
            other_jpeg = root / "IMG_0002.JPG"
            copy_jpeg = root / "IMG_0001 copy.JPG"
            for path in (raw, same_jpeg, other_jpeg, copy_jpeg):
                path.write_bytes(b"x")
            stamp = time.time()
            for path in (raw, same_jpeg, other_jpeg, copy_jpeg):
                os.utime(path, (stamp, stamp))

            merged = engine._merge_raw_jpeg_by_timestamp(
                {
                    "raw": [str(raw)],
                    "same": [str(same_jpeg)],
                    "other": [str(other_jpeg)],
                    "copy": [str(copy_jpeg)],
                }
            )
            self.assertIn(str(other_jpeg), merged["other"])
            self.assertEqual(len(merged["other"]), 1)
            combined = [files for files in merged.values() if str(raw) in files][0]
            self.assertIn(str(same_jpeg), combined)
            self.assertIn(str(copy_jpeg), combined)
            self.assertNotIn(str(other_jpeg), combined)


if __name__ == "__main__":
    unittest.main()
