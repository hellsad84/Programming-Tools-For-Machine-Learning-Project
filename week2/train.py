"""Owner: Teammate 3. Guide sections 17-19.

Imports of the data/model components belong inside run(), allowing --help and
train_step fixture tests before the other owners deliver their components.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_step(model, batch, optimizer, clip_norm):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    hidden = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
    log_probs = model.generator(hidden)
    summed = F.nll_loss(
        log_probs.reshape(-1, log_probs.size(-1)),
        batch.tgt_y.reshape(-1), ignore_index=0, reduction="sum")
    count = int((batch.tgt_y != 0).sum())
    if count == 0 or not bool(torch.isfinite(summed)):
        raise ValueError("Invalid loss or no non-PAD target tokens")
    (summed / count).backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(summed.detach()), count


def save_checkpoint(path, model, optimizer, metadata):
    """Save trusted local state for evaluation; no resume feature is promised."""
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                **metadata}, path)


def run(args):
    import platform
    from week2.data_pipeline import load_artifacts, make_loader
    from week2.model import build_model
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.steps is not None:
        cfg["steps"] = args.steps
    positive = ("steps", "batch_size", "cpu_threads", "log_every", "save_every")
    if any(cfg[name] < 1 for name in positive):
        raise ValueError("Steps, batch size, threads and logging intervals must be positive")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new, empty run output directory")
    tokenizer, manifest = load_artifacts(cfg["data_dir"])
    torch.set_num_threads(cfg["cpu_threads"])
    seed_everything(cfg["seed"])
    model = build_model(tokenizer.get_piece_size(), args.heads, cfg).cpu()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]), eps=cfg["eps"],
        weight_decay=cfg["weight_decay"])
    loader = make_loader(cfg["data_dir"], "train", cfg["batch_size"],
                         cfg["seed"], True)
    iterator = iter(loader)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(
        json.dumps({**cfg, "heads": args.heads, "device": "cpu"}, indent=2),
        encoding="utf-8")
    environment = {"python": platform.python_version(),
                   "platform": platform.platform(), "torch": str(torch.__version__)}
    (out / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8")
    fingerprint = hashlib.sha256(
        (Path(cfg["data_dir"]) / "manifest.json").read_bytes()).hexdigest()
    metadata = {"config": cfg, "heads": args.heads,
                "vocab_size": tokenizer.get_piece_size(),
                "manifest_sha256": fingerprint,
                "tokenizer_sha256": manifest["files"]["tokenizer.model"]}
    total_tokens = total_examples = interval_tokens = 0
    interval_nll = training_seconds = 0.0
    order = hashlib.sha256()
    fields = ["step", "train_nll", "tokens_seen", "examples_seen", "train_seconds"]
    with (out / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(1, cfg["steps"] + 1):
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            summed, count = train_step(model, batch, optimizer, cfg["clip_norm"])
            training_seconds += time.perf_counter() - start
            order.update(("\n".join(batch.ids) + "\n--batch--\n").encode())
            total_tokens += count
            total_examples += len(batch.ids)
            interval_tokens += count
            interval_nll += summed
            if step % cfg["log_every"] == 0 or step == cfg["steps"]:
                record = {"step": step, "train_nll": interval_nll / interval_tokens,
                          "tokens_seen": total_tokens, "examples_seen": total_examples,
                          "train_seconds": training_seconds}
                writer.writerow(record)
                f.flush()
                print(record)
                interval_nll = 0.0
                interval_tokens = 0
            metadata.update(step=step, tokens_seen=total_tokens,
                            examples_seen=total_examples, batch_order_sha256=order.hexdigest(),
                            train_seconds=training_seconds)
            if step % cfg["save_every"] == 0:
                save_checkpoint(out / f"step_{step}.pt", model, optimizer, metadata)
    save_checkpoint(out / "final.pt", model, optimizer, metadata)
    print("Saved", out / "final.pt")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/week2_cpu.json")
    parser.add_argument("--heads", type=int, choices=[1, 8], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
