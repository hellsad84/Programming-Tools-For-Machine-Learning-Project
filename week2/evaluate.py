"""Owner: Teammate 3. Guide sections 20-21. Validation only during Week 2."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU


def corpus_bleu(predictions, references):
    if not predictions or len(predictions) != len(references):
        raise ValueError("Prediction/reference counts must match and be nonzero")
    metric = BLEU(tokenize="13a", lowercase=False)
    result = metric.corpus_score(predictions, [references])
    return {"bleu": result.score, "signature": str(metric.get_signature())}


def run(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new evaluation output directory")
    from week2.data_pipeline import load_artifacts, make_loader
    from week2.model import build_model
    from week2.decode import translate_batch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    torch.set_num_threads(cfg["cpu_threads"])
    tokenizer, manifest = load_artifacts(cfg["data_dir"])
    fingerprint = hashlib.sha256(
        (Path(cfg["data_dir"]) / "manifest.json").read_bytes()).hexdigest()
    if fingerprint != checkpoint["manifest_sha256"]:
        raise ValueError("Data manifest differs from training")
    if manifest["files"]["tokenizer.model"] != checkpoint["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from training")
    model = build_model(checkpoint["vocab_size"], checkpoint["heads"], cfg).cpu()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = make_loader(cfg["data_dir"], "valid", cfg["eval_batch_size"],
                         cfg["seed"], False)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    target_count = min(args.limit or len(loader.dataset), len(loader.dataset))
    predictions, references, ids, sources = [], [], [], []
    total_nll = total_tokens = decode_seconds = 0.0
    with torch.no_grad():
        for batch in loader:
            take = min(len(batch.ids), target_count - len(ids))
            if take <= 0:
                break
            hidden = model(batch.src[:take], batch.tgt[:take],
                           batch.src_mask[:take], batch.tgt_mask[:take])
            scores = model.generator(hidden)
            total_nll += float(F.nll_loss(
                scores.reshape(-1, scores.size(-1)), batch.tgt_y[:take].reshape(-1),
                ignore_index=0, reduction="sum"))
            total_tokens += int((batch.tgt_y[:take] != 0).sum())
            start = time.perf_counter()
            predictions.extend(translate_batch(
                model, batch.src[:take], batch.src_mask[:take],
                tokenizer, cfg["max_new_tokens"]))
            decode_seconds += time.perf_counter() - start
            references.extend(batch.references[:take])
            sources.extend(batch.source_texts[:take])
            ids.extend(batch.ids[:take])
    if len(ids) != target_count or len(predictions) != len(references):
        raise ValueError("Evaluation row counts do not match")
    if total_tokens <= 0:
        raise ValueError("No evaluation labels")
    out.mkdir(parents=True, exist_ok=True)
    for name, lines in [("valid.pred.de", predictions), ("valid.ref.de", references),
                        ("valid.ids.txt", ids), ("valid.src.en", sources)]:
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics = {
        **corpus_bleu(predictions, references),
        "validation_nll": total_nll / total_tokens,
        "evaluated_pairs": len(ids), "split": "valid",
        "timing_subset_only": args.limit is not None,
        "heads": checkpoint["heads"], "steps": checkpoint["step"],
        "seed": cfg["seed"], "decode_seconds": decode_seconds,
        "train_seconds": checkpoint["train_seconds"],
        "tokens_seen": checkpoint["tokens_seen"],
        "examples_seen": checkpoint["examples_seen"],
        "batch_order_sha256": checkpoint["batch_order_sha256"],
        "manifest_sha256": checkpoint["manifest_sha256"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "max_new_tokens": cfg["max_new_tokens"]
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["valid"], default="valid")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
