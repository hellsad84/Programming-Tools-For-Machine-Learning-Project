"""Padded batching and artifact loading for Week 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from week2.prepare_data import normalize

PAD, BOS, EOS, UNK = 0, 1, 2, 3

CHUNK = 1 << 20


def subsequent_mask(size):
    attention_shape = (1, size, size)
    mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(torch.uint8)
    return mask == 0


class Batch:
    """One collated batch; the decoder input drops the last target token and the
    labels drop the first one."""

    def __init__(self, src, tgt=None, pad=PAD):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).sum()
        self.ids = []
        self.source_texts = []
        self.references = []

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


def encode_text(text, tokenizer):
    ids = tokenizer.encode(normalize(text))
    return torch.tensor([BOS, *ids, EOS], dtype=torch.long)


def collate_rows(rows, tokenizer):
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    source = [encode_text(r["src"], tokenizer) for r in rows]
    target = [encode_text(r["tgt"], tokenizer) for r in rows]
    src = pad_sequence(source, batch_first=True, padding_value=PAD)
    tgt = pad_sequence(target, batch_first=True, padding_value=PAD)
    batch = Batch(src, tgt, pad=PAD)
    batch.ids = [r["id"] for r in rows]
    batch.source_texts = [r["src"] for r in rows]
    batch.references = [r["tgt"] for r in rows]
    return batch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifacts(data_dir):
    """Load the tokenizer and manifest, checking every recorded hash."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}; run week2.prepare_data first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared artifact: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"Artifact hash mismatch for {name}: {actual} != {expected}")
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(data_dir / "tokenizer.model"))
    if tokenizer.get_piece_size() != manifest["vocab_size"]:
        raise ValueError(
            f"Tokenizer has {tokenizer.get_piece_size()} pieces but the manifest "
            f"records {manifest['vocab_size']}")
    if (tokenizer.pad_id(), tokenizer.bos_id(),
            tokenizer.eos_id(), tokenizer.unk_id()) != (PAD, BOS, EOS, UNK):
        raise ValueError("Tokenizer special ids are not 0, 1, 2, 3")
    return tokenizer, manifest


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_loader(data_dir, split, batch_size, seed, shuffle):
    data_dir = Path(data_dir)
    tokenizer, manifest = load_artifacts(data_dir)
    rows = read_jsonl(data_dir / f"{split}.jsonl")
    if not rows:
        raise ValueError(f"Split {split!r} is empty in {data_dir}")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=lambda batch: collate_rows(batch, tokenizer),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Week 2 padded batching utilities")
    parser.add_argument("--check", action="store_true",
                        help="load artifacts and print one batch")
    parser.add_argument("--data-dir", default="data/week2")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return 1
    tokenizer, manifest = load_artifacts(args.data_dir)
    print("counts:", manifest["counts"])
    print("vocab_size:", tokenizer.get_piece_size())
    print("special_ids:", manifest["special_ids"])
    loader = make_loader(args.data_dir, "train", args.batch_size,
                         manifest["config"]["subset_seed"], False)
    batch = next(iter(loader))
    print("src", tuple(batch.src.size()), "tgt", tuple(batch.tgt.size()),
          "tgt_y", tuple(batch.tgt_y.size()))
    print("src_mask", tuple(batch.src_mask.size()),
          "tgt_mask", tuple(batch.tgt_mask.size()))
    print("ids:", batch.ids[:2])
    print("ntokens:", int(batch.ntokens))
    if int(batch.ntokens) <= 0:
        raise ValueError("Batch has no non-PAD target tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
