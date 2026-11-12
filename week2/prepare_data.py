"""Week 2 data preparation: Multi30k download, selection, tokenizer and frozen package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import unicodedata
import urllib.request
from pathlib import Path

SPLITS = ("train", "valid", "test")
LANGS = ("en", "de")

# Upstream stems under data/task1/raw: "valid" is stored as val, and the reserved
# test split keeps its Flickr-specific name.
SOURCE_STEMS = {
    "train": "train",
    "valid": "val",
    "test": "test_2016_flickr",
}

CHUNK = 1 << 20
USER_AGENT = "week2-data-preparation/1.0"


def normalize(text):
    return unicodedata.normalize("NFC", text).strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_path(config, split, lang):
    return Path(config["raw_dir"]) / f"{SOURCE_STEMS[split]}.{lang}.gz"


def source_url(config, split, lang):
    return str(config["source_url_template"]).format(
        revision=config["revision"],
        stem=SOURCE_STEMS[split],
        lang=lang,
    )


def count_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def download_raw(config):
    """Download the six upstream files once and write the receipt."""
    raw_dir = Path(config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        for lang in LANGS:
            destination = raw_path(config, split, lang)
            url = source_url(config, split, lang)
            if not destination.is_file():
                _fetch(url, destination)
            files[destination.name] = {
                "url": url,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "lines": count_lines(destination),
            }
    for split in SPLITS:
        en_lines = files[raw_path(config, split, "en").name]["lines"]
        de_lines = files[raw_path(config, split, "de").name]["lines"]
        if en_lines != de_lines:
            raise ValueError(
                f"{split}: English has {en_lines} lines but German has "
                f"{de_lines}; the split is not aligned")
    receipt = {
        "revision": config["revision"],
        "source_url_template": config["source_url_template"],
        "files": files,
    }
    (raw_dir / "download_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def _fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
    temporary.replace(destination)


def read_pairs(raw_dir, split):
    """Read one aligned split as rows of id, src and tgt."""
    stem = SOURCE_STEMS[split]
    raw_dir = Path(raw_dir)
    sources = _read_lines(raw_dir / f"{stem}.en.gz")
    targets = _read_lines(raw_dir / f"{stem}.de.gz")
    if len(sources) != len(targets):
        raise ValueError(
            f"{split}: {len(sources)} English lines and {len(targets)} German lines")
    return [
        {"id": f"{split}-{index:05d}", "src": src, "tgt": tgt}
        for index, (src, tgt) in enumerate(zip(sources, targets))
    ]


def _read_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def select_training(train, valid, test, config):
    held_sources = {normalize(r["src"]) for r in valid + test}
    seen_pairs = set()
    eligible, removed = [], []
    for row in sorted(train, key=lambda r: r["id"]):
        pair = (normalize(row["src"]), normalize(row["tgt"]))
        if pair[0] in held_sources:
            reason = "source_overlap_with_held_out"
        elif pair in seen_pairs:
            reason = "duplicate_training_pair"
        else:
            seen_pairs.add(pair)
            eligible.append(row)
            continue
        removed.append({"id": row["id"], "reason": reason})
    random.Random(config["subset_seed"]).shuffle(eligible)
    chosen = eligible[:config["subset_size"]]
    overlap = ({normalize(r["src"]) for r in valid}
               & {normalize(r["src"]) for r in test})
    audit = {
        "removed": removed,
        "eligible_count": len(eligible),
        "candidate_ids": [r["id"] for r in chosen],
        "valid_test_source_overlap_count": len(overlap)
    }
    return chosen, audit


def train_tokenizer(candidate_rows, config):
    import sentencepiece as spm
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    corpus = out / "tokenizer_input.txt"
    lines = [normalize(r[key]) for r in candidate_rows
             for key in ("src", "tgt")]
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(out / "tokenizer"),
        model_type="bpe", vocab_size=config["vocab_size"],
        character_coverage=1.0, pad_id=0, bos_id=1, eos_id=2,
        unk_id=3, hard_vocab_limit=False,
        shuffle_input_sentence=False, input_sentence_size=0,
        num_threads=1, normalization_rule_name="identity")
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(out / "tokenizer.model"))
    assert (tokenizer.pad_id(), tokenizer.bos_id(),
            tokenizer.eos_id(), tokenizer.unk_id()) == (0, 1, 2, 3)
    return tokenizer


def prepare(config):
    out = Path(config["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output is not empty; use a new output_dir")
    receipt = download_raw(config)
    rows = {s: read_pairs(config["raw_dir"], s) for s in SPLITS}
    candidates, audit = select_training(
        rows["train"], rows["valid"], rows["test"], config)
    if not candidates:
        raise ValueError("No eligible training rows")
    tokenizer = train_tokenizer(candidates, config)
    kept, long_ids = [], []
    for row in candidates:
        lengths = [len(tokenizer.encode(normalize(row[k]))) + 2
                   for k in ("src", "tgt")]
        if max(lengths) <= config["max_train_tokens"]:
            kept.append(row)
        else:
            long_ids.append(row["id"])
    if not kept:
        raise ValueError("Length filter removed all training rows")
    rows["train"] = kept
    audit["long_training_ids"] = long_ids
    audit["final_training_ids"] = [r["id"] for r in kept]
    for split, records in rows.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    names = ["train.jsonl", "valid.jsonl", "test.jsonl",
             "tokenizer.model", "tokenizer.vocab", "audit.json"]
    manifest = {
        "config": config, "source_files": receipt,
        "counts": {s: len(r) for s, r in rows.items()},
        "vocab_size": tokenizer.get_piece_size(),
        "special_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
        "files": {name: sha256(out / name) for name in names}
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared:", manifest["counts"], "vocab:", manifest["vocab_size"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare the Week 2 Multi30k package")
    parser.add_argument("--config", required=True, help="path to week2_data.json")
    parser.add_argument("--download-only", action="store_true",
                        help="download and verify the raw files, then stop")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.download_only:
        receipt = download_raw(config)
        print("Downloaded and verified:", len(receipt["files"]), "files")
        return 0
    prepare(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
