"""Teammate 3 setup check. Default checks your files; --full checks deliveries too."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--config", default="configs/week2_cpu.json")
    args = parser.parse_args()
    problems = []
    packages = ["torch", "numpy", "sacrebleu"]
    owned = [args.config, "week2/train.py", "week2/evaluate.py",
             "tests/test_week2_training.py"]
    if args.full:
        packages.append("sentencepiece")
        owned += ["starter_core.py", "week2/prepare_data.py", "week2/data_pipeline.py",
                  "week2/model.py", "week2/decode.py"]
    for name in packages:
        if importlib.util.find_spec(name) is None:
            problems.append(f"Missing Python package: {name}")
    for name in owned:
        p = Path(name)
        if not p.is_file():
            problems.append(f"Missing file: {name}")
            continue
        if p.suffix == ".py":
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
                incomplete = any(isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                                 and isinstance(node.exc.func, ast.Name)
                                 and node.exc.func.id == "NotImplementedError"
                                 for node in ast.walk(tree))
                if incomplete:
                    problems.append(f"Unfinished template function in: {name}")
            except SyntaxError as error:
                problems.append(f"Syntax error in {name}: {error}")
    if args.full and Path(args.config).is_file():
        cfg = json.loads(Path(args.config).read_text())
        root = Path(cfg["data_dir"])
        path = root / "manifest.json"
        if not path.is_file():
            problems.append(f"Missing data delivery: {path}")
        else:
            manifest = json.loads(path.read_text())
            for name, expected in manifest["files"].items():
                artifact = root / name
                if not artifact.is_file():
                    problems.append(f"Missing data artifact: {artifact}")
                elif hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                    problems.append(f"Changed data artifact: {artifact}")
    if problems:
        print("Not ready yet:")
        for problem in problems:
            print(" -", problem)
        raise SystemExit(1)
    print("Setup checks passed. Run your tests next; this does not prove training quality.")


if __name__ == "__main__":
    main()
