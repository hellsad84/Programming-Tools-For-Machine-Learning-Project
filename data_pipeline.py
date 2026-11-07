import torch
from torch.nn.utils.rnn import pad_sequence
from starter_core import Batch

PAIRS = [("hello", "hallo"), ("i am here", "ich bin hier")]
PAD, START, END = 0, 1, 2

def make_batch():
    words = sorted({word for pair in PAIRS for text in pair
                    for word in text.split()})
    vocab = {word: i + 3 for i, word in enumerate(words)}

    def encode(text):
        ids = [vocab[word] for word in text.split()]
        return torch.tensor([START, *ids, END])

    src, tgt = zip(*PAIRS)
    def stack(sentences):
        rows = [encode(text) for text in sentences]
        return pad_sequence(rows, batch_first=True, padding_value=PAD)

    return Batch(stack(src), stack(tgt), pad=PAD), len(vocab) + 3

if __name__ == "__main__":
    batch, size = make_batch()
    print(PAIRS[0], "vocab size:", size)
    print("source shape:", tuple(batch.src.shape))
    print("visible source tokens:", batch.src_mask[0].tolist())
    print("German input:", batch.tgt[0].tolist())
    print("German labels:", batch.tgt_y[0].tolist())
