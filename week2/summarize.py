"""Check paired runs and write a small results table. Owner: Teammate 3."""
import argparse
import json
import math
from pathlib import Path


def summarize(first, second, output):
    roots = [Path(first), Path(second)]
    rows = [json.loads((root / "eval/metrics.json").read_text()) for root in roots]
    configs = [json.loads((root / "resolved_config.json").read_text()) for root in roots]
    if [row["heads"] for row in rows] != [1, 8]:
        raise ValueError("Pass the 1-head run first and the 8-head run second")
    for row in rows:
        if row["timing_subset_only"] or row["split"] != "valid":
            raise ValueError("Use full validation results, not timing subsets")
        if not all(math.isfinite(row[k]) for k in ("bleu", "validation_nll")):
            raise ValueError("Metrics must be finite")
    keys = ["steps", "seed", "evaluated_pairs", "tokens_seen", "examples_seen",
            "batch_order_sha256", "manifest_sha256", "parameter_count",
            "max_new_tokens", "signature"]
    for key in keys:
        if rows[0][key] != rows[1][key]:
            raise ValueError(f"Paired run mismatch: {key}")
    for cfg in configs:
        cfg.pop("heads", None)
    if configs[0] != configs[1]:
        raise ValueError("Resolved training configurations differ beyond head count")
    lines = ["# Week 2 measured validation results", "",
             "| Heads | Updates | Validation NLL | BLEU | Train seconds | Decode seconds |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['heads']} | {row['steps']} | {row['validation_nll']:.4f} | "
                     f"{row['bleu']:.2f} | {row['train_seconds']:.2f} | {row['decode_seconds']:.2f} |")
    lines += ["", f"BLEU difference (8 minus 1): {rows[1]['bleu'] - rows[0]['bleu']:.2f}",
              f"Evaluated pairs: {rows[0]['evaluated_pairs']}",
              f"SacreBLEU signature: `{rows[0]['signature']}`", "",
              "One CPU pilot with one seed; this does not establish a general winner.",
              "", "Record the machine, code commit and artifact locations in the component report."]
    output = Path(output)
    if output.exists():
        raise FileExistsError("Summary already exists; use a new output filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", default="runs/week2/h1_s11")
    parser.add_argument("--eight", default="runs/week2/h8_s11")
    parser.add_argument("--out", default="results/week2/summary.md")
    args = parser.parse_args()
    summarize(args.one, args.eight, args.out)
