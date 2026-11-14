"""Owner: Teammate 2. Guide section 13. Reuse credited starter_core.py."""
import argparse
import torch
from starter_core import Batch, make_model

FIXTURE_CONFIG = {"layers": 1, "d_model": 128, "d_ff": 256, "dropout": 0.1}



def build_model(vocab_size, heads, config):
    if heads not in (1, 8):
        raise ValueError(
            "Week 2 compares heads=1 and heads=8"
        )

    if config["d_model"] % heads != 0:
        raise ValueError(
            "d_model must be divisible by heads"
        )

    model = make_model(
        vocab_size,
        vocab_size,
        N=config["layers"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        h=heads,
        dropout=config["dropout"]
    )

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = config["dropout"]

    return model



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    torch.set_num_threads(2)
    src = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])
    tgt = torch.tensor([[1, 7, 8, 2], [1, 9, 2, 0]])
    batch = Batch(src, tgt, pad=0)
    counts = []
    for heads in (1, 8):
        torch.manual_seed(11)
        model = build_model(16, heads, FIXTURE_CONFIG)
        model.eval()
        with torch.no_grad():
            hidden = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            scores = model.generator(hidden)
        assert tuple(hidden.shape) == (2, 3, 128)
        assert tuple(scores.shape) == (2, 3, 16)
        assert torch.allclose(scores.exp().sum(-1), torch.ones(2, 3), atol=1e-5)
        counts.append(sum(p.numel() for p in model.parameters()))
        print("heads", heads, "hidden", tuple(hidden.shape), "parameters", counts[-1])
    assert counts[0] == counts[1]


if __name__ == "__main__":
    main()
