import sys
import torch
import torch.nn.functional as F
from data_pipeline import PAD, make_batch
from model import WIDTH, build_model

def run(heads, seed):
    torch.manual_seed(seed)
    batch, size = make_batch()
    print("target_tokens", batch.ntokens.item())
    model = build_model(size, heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    hidden = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
    scores = model.generator(hidden)
    loss = F.nll_loss(scores.reshape(-1, size),
                      batch.tgt_y.reshape(-1), ignore_index=PAD)
    assert torch.isfinite(loss), "Loss must be finite"
    before = model.generator.proj.weight.detach().clone()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.generator.proj.weight)
    print(f"seed={seed} heads={heads} width={WIDTH} updates=1 "
          f"loss_before_update={loss.item():.4f}")

if __name__ == "__main__":
    import platform
    print("Python", platform.python_version(), "torch", torch.__version__)
    run(int(sys.argv[1]), int(sys.argv[2]))
