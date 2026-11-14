"""Week 2 data tests (walkthrough D1-D5)."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from week2.prepare_data import normalize, prepare, select_training, sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "week2"
RAW_DIR = REPO_ROOT / "data" / "raw" / "multi30k"
CONFIG_PATH = REPO_ROOT / "configs" / "week2_data.json"

try:
    from week2.data_pipeline import BOS, EOS, PAD, collate_rows, load_artifacts
    HAVE_TORCH = True
except Exception:
    BOS, EOS, PAD = 1, 2, 0
    collate_rows = load_artifacts = None
    HAVE_TORCH = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "torch/sentencepiece are not installed")
needs_package = unittest.skipUnless(
    (DATA_DIR / "manifest.json").is_file() and (DATA_DIR / "train.jsonl").is_file(),
    "no prepared package in data/week2; run week2.prepare_data first")
needs_raw = unittest.skipUnless(
    (RAW_DIR / "train.en.gz").is_file(),
    "no raw corpus in data/raw/multi30k; run week2.prepare_data --download-only first")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class WordTokenizer:
    """Stand-in for SentencePiece: one id per whitespace token."""

    def __init__(self):
        self.vocab = {}
        self.next_id = 4

    def encode(self, text):
        ids = []
        for word in normalize(text).split():
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.next_id += 1
            ids.append(self.vocab[word])
        return ids


def tiny_rows():
    return (
        [
            {"id": "train-00000", "src": "a dog runs", "tgt": "ein Hund läuft"},
            {"id": "train-00001", "src": "a dog runs", "tgt": "ein Hund läuft"},
            {"id": "train-00002", "src": "a cat sleeps", "tgt": "eine Katze schläft"},
            {"id": "train-00003", "src": "a bird sings", "tgt": "ein Vogel singt"},
        ],
        [{"id": "valid-00000", "src": "a cat sleeps", "tgt": "eine Katze schläft"}],
        [{"id": "test-00000", "src": "a train arrives", "tgt": "ein Zug kommt an"}],
    )


class SelectionTests(unittest.TestCase):

    def test_duplicates_and_held_out_rows_are_removed(self):
        train, valid, test = tiny_rows()
        chosen, audit = select_training(train, valid, test,
                                        {"subset_size": 10, "subset_seed": 11})
        chosen_ids = sorted(row["id"] for row in chosen)
        self.assertEqual(chosen_ids, ["train-00000", "train-00003"])
        self.assertEqual(
            {(item["id"], item["reason"]) for item in audit["removed"]},
            {("train-00001", "duplicate_training_pair"),
             ("train-00002", "source_overlap_with_held_out")})
        self.assertEqual(audit["eligible_count"], 2)
        self.assertEqual(audit["valid_test_source_overlap_count"], 0)

    def test_subset_seed_is_reproducible(self):
        train, valid, test = tiny_rows()
        config = {"subset_size": 1, "subset_seed": 11}
        first, _ = select_training(train, valid, test, config)
        second, _ = select_training(train, valid, test, dict(config))
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])
        self.assertEqual(len(first), 1)

    def test_training_sources_do_not_overlap_held_out_sources(self):
        train, valid, test = tiny_rows()
        chosen, _ = select_training(train, valid, test,
                                    {"subset_size": 10, "subset_seed": 11})
        held = {normalize(r["src"]) for r in valid + test}
        kept = {normalize(r["src"]) for r in chosen}
        self.assertEqual(kept & held, set())

    @needs_package
    def test_prepared_package_has_no_held_out_overlap(self):
        train = read_jsonl(DATA_DIR / "train.jsonl")
        held = {normalize(r["src"])
                for split in ("valid", "test")
                for r in read_jsonl(DATA_DIR / f"{split}.jsonl")}
        kept = {normalize(r["src"]) for r in train}
        self.assertEqual(kept & held, set())


class BatchTests(unittest.TestCase):

    rows = [
        {"id": "b-1", "src": "a b c d", "tgt": "x"},
        {"id": "b-2", "src": "a", "tgt": "x y z"},
    ]

    @needs_torch
    def test_padding_is_zero_and_only_on_the_right(self):
        batch = collate_rows(self.rows, WordTokenizer())
        src = batch.src.tolist()
        self.assertEqual(len(src[0]), len(src[1]))
        for row in src:
            self.assertEqual(row[0], BOS)
            last = max(index for index, value in enumerate(row) if value != PAD)
            self.assertEqual(row[last], EOS)
            self.assertTrue(all(value == PAD for value in row[last + 1:]))
        for row, row_mask in zip(src, batch.src_mask.tolist()):
            for value, keep in zip(row, row_mask[0]):
                self.assertEqual(bool(keep), value != PAD)

    @needs_torch
    def test_labels_are_shifted_and_pad_is_not_counted(self):
        batch = collate_rows(self.rows, WordTokenizer())
        expected_tokens = int((batch.tgt_y != PAD).sum())
        self.assertGreater(expected_tokens, 0)
        self.assertEqual(int(batch.ntokens), expected_tokens)
        for labels, inputs in zip(batch.tgt_y.tolist(), batch.tgt.tolist()):
            self.assertEqual(labels[:-1], inputs[1:])
            self.assertEqual(labels[0], inputs[1])
            last = max(index for index, value in enumerate(labels) if value != PAD)
            self.assertEqual(labels[last], EOS)

    @needs_torch
    def test_target_mask_hides_future_keys(self):
        batch = collate_rows(self.rows, WordTokenizer())
        width = batch.tgt.size(-1)
        self.assertEqual(tuple(batch.tgt_mask.size()), (len(self.rows), width, width))
        for sequence in batch.tgt_mask.tolist():
            for position, row in enumerate(sequence):
                for key, allowed in enumerate(row):
                    if key > position:
                        self.assertFalse(allowed)


class PackageTests(unittest.TestCase):

    @needs_package
    @needs_raw
    def test_second_prepare_matches_the_first(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        first = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
        first_audit = json.loads((DATA_DIR / "audit.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            second_dir = Path(temporary) / "week2"
            rerun = copy.deepcopy(config)
            rerun["output_dir"] = str(second_dir)
            prepare(rerun)
            second = json.loads((second_dir / "manifest.json").read_text(encoding="utf-8"))
            second_audit = json.loads((second_dir / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(first["counts"], second["counts"])
            self.assertEqual(first["vocab_size"], second["vocab_size"])
            self.assertEqual(first_audit["final_training_ids"],
                             second_audit["final_training_ids"])
            for split in ("train", "valid", "test"):
    first_text = (DATA_DIR / f"{split}.jsonl").read_text(
        encoding="utf-8"
    )
    second_text = (second_dir / f"{split}.jsonl").read_text(
        encoding="utf-8"
    )
    self.assertEqual(
        first_text,
        second_text,
        msg=f"{split}: regenerated dataset content differs",
    )

    @needs_package
    def test_tampered_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_dir = Path(temporary) / "week2"
            shutil.copytree(DATA_DIR, copy_dir)
            load_artifacts(copy_dir)
            manifest = json.loads((copy_dir / "manifest.json").read_text(encoding="utf-8"))
            victim = copy_dir / sorted(manifest["files"])[0]
            tampered = bytearray(victim.read_bytes())
            tampered[-1] ^= 0xFF
            victim.write_bytes(bytes(tampered))
            with self.assertRaises(ValueError):
                load_artifacts(copy_dir)


if __name__ == "__main__":
    unittest.main()
