from starter_core import make_model

WIDTH = 128

def build_model(vocab_size, heads):
    return make_model(vocab_size, vocab_size, N=1, d_model=WIDTH,
                      d_ff=256, h=heads, dropout=0.0)


if __name__ == "__main__":
    from data_pipeline import make_batch
    batch, size = make_batch()
    for heads in (1, 8):
        model = build_model(size, heads)
        hidden = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        assert hidden.shape == (*batch.tgt.shape, WIDTH)
        print("heads:", heads, "width per head:", WIDTH // heads, tuple(hidden.shape))
