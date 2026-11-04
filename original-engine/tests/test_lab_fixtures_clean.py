"""The repository is pushed to GitHub every session, so no file in it may be (or contain a row of) the IFBench test set (decision D10).

``scripts/lab_pins.json`` holds only hashes: the sha256 of the known test files and sample outputs, their sizes, and the sha256 of every test
prompt. This test walks the files git would publish (tracked, plus untracked and not ignored) and fails when

* a file's size and sha256 equal a pinned test or sample-output file, or
* a JSON or JSONL file under tests/, scripts/, nightshift/ or docs/ has an object whose ``prompt`` field hashes to a pinned test prompt.
"""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "scripts" / "lab_pins.json").read_text())["forbidden_in_repository"]
FILE_HASHES = {h for group in ("ifbench_test_file_sha256", "sample_output_file_sha256") for h in PINS[group].values()}
SIZES = set(PINS["file_sizes_bytes"])
PROMPT_HASHES = set(PINS["prompt_sha256"])
ROW_DIRS = ("tests", "scripts", "nightshift", "docs")
SKIP_DIRS = {".git", "__pycache__", "node_modules"}
MAX_ROW_SCAN_BYTES = 64 * 1024 * 1024


def publishable_files(root: Path) -> list:
    """Files git would add: tracked and untracked-but-not-ignored. Falls back to a plain walk without git."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                             capture_output=True, check=True, timeout=60).stdout
        names = [n for n in out.decode("utf-8", "surrogateescape").split("\0") if n]
        return [root / n for n in names if (root / n).is_file() and not (root / n).is_symlink()]
    except (OSError, subprocess.SubprocessError):
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            found += [Path(dirpath) / f for f in filenames if not (Path(dirpath) / f).is_symlink()]
        return found


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompts_in(path: Path) -> list:
    """Every ``prompt`` string of every JSON object in a .json or .jsonl file (empty if the file is not JSON)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    docs = []
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            try:
                docs.append(json.loads(line))
            except ValueError:
                continue
    else:
        try:
            docs.append(json.loads(text))
        except ValueError:
            return []
    found, stack = [], docs
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if isinstance(x.get("prompt"), str):
                found.append(x["prompt"])
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return found


def violations(files: list, root: Path) -> list:
    bad = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if size in SIZES and sha256(f) in FILE_HASHES:
            bad.append(f"{rel}: is a copy of an IFBench test or sample-output file")
            continue
        if f.suffix in (".json", ".jsonl") and rel.split("/")[0] in ROW_DIRS and size <= MAX_ROW_SCAN_BYTES:
            for p in prompts_in(f):
                if hashlib.sha256(p.encode()).hexdigest() in PROMPT_HASHES or hashlib.sha256(p.strip().encode()).hexdigest() in PROMPT_HASHES:
                    bad.append(f"{rel}: holds a row whose prompt is an IFBench test prompt")
                    break
    return bad


class FixturesCleanTests(unittest.TestCase):
    def test_pins_are_present_and_cover_the_known_files(self):
        self.assertGreaterEqual(len(FILE_HASHES), 5)
        self.assertGreaterEqual(len(PROMPT_HASHES), 300)
        self.assertTrue(all(len(h) == 64 for h in FILE_HASHES | PROMPT_HASHES))

    def test_the_repository_holds_no_ifbench_test_data(self):
        files = publishable_files(ROOT)
        self.assertGreater(len(files), 50)
        self.assertEqual(violations(files, ROOT), [])

    def test_fixtures_exist_and_are_scanned(self):
        files = publishable_files(ROOT)
        fixture = ROOT / "tests" / "fixtures" / "lab" / "ifbench_mini" / "rows.jsonl"
        self.assertIn(fixture, files)
        self.assertGreater(len(prompts_in(fixture)), 100)

    def test_the_check_catches_a_copied_file_and_a_copied_row(self):
        # Synthetic stand-ins: pins are swapped for the hashes of files this test writes, so no real test data is needed.
        global FILE_HASHES, SIZES, PROMPT_HASHES
        saved = (FILE_HASHES, SIZES, PROMPT_HASHES)
        try:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "tests").mkdir()
                secret = '{"key": "0", "prompt": "a made up pretend test prompt number one", "instruction_id_list": [], "kwargs": []}\n'
                copy_ = root / "tests" / "copy.jsonl"
                copy_.write_text(secret)
                reordered = root / "tests" / "other.jsonl"
                reordered.write_text('{"prompt": "a made up pretend test prompt number one", "extra": 1}\n')
                clean = root / "tests" / "clean.jsonl"
                clean.write_text('{"prompt": "something entirely different"}\n')
                nested = root / "tests" / "nested.json"
                nested.write_text(json.dumps({"rows": [{"prompt": "a made up pretend test prompt number one "}]}))
                FILE_HASHES = {sha256(copy_)}
                SIZES = {copy_.stat().st_size}
                PROMPT_HASHES = {hashlib.sha256(b"a made up pretend test prompt number one").hexdigest()}
                found = violations([copy_, reordered, clean, nested], root)
                self.assertEqual(sorted(x.split(":")[0] for x in found), ["tests/copy.jsonl", "tests/nested.json", "tests/other.jsonl"])
        finally:
            FILE_HASHES, SIZES, PROMPT_HASHES = saved


if __name__ == "__main__":
    unittest.main()
