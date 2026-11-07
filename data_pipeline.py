import torch
from torch.nn.utils.rnn import pad_sequence
from starter_core import Batch

PAIRS = [("hello", "hallo"), ("i am here", "ich bin hier")]
PAIRS.append(("good night", "gute nacht"))
PAIRS.append(("good morning", "guten morgen"))
PAIRS.append(("good evening", "guten abend"))
PAIRS.append(("thank you", "danke"))
PAIRS.append(("see you soon", "bis bald"))
PAIRS.append(("welcome", "willkommen"))
PAIRS.append(("i am tired", "ich bin müde"))
PAIRS.append(("i am hungry", "ich bin hungrig"))
PAIRS.append(("i am happy", "ich bin glücklich"))
PAIRS.append(("i am ready", "ich bin bereit"))
PAIRS.append(("i need help", "ich brauche hilfe"))
PAIRS.append(("i like music", "ich mag musik"))
PAIRS.append(("i love you", "ich liebe dich"))
PAIRS.append(("the sun shines", "die sonne scheint"))
PAIRS.append(("the water is cold", "das wasser ist kalt"))
PAIRS.append(("the food is good", "das essen ist gut"))
PAIRS.append(("where are you", "wo bist du"))
PAIRS.append(("what is your name", "wie heißt du"))
PAIRS.append(("my name is anna", "ich heiße anna"))
PAIRS.append(("we are learning german", "wir lernen deutsch"))
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

        batch = Batch(stack(src), stack(tgt), pad=PAD)
    batch.vocab = vocab
    return batch, len(vocab) + 3

if __name__ == "__main__":
    batch, size = make_batch()
    print(PAIRS[0], "vocab size:", size)
    print("source shape:", tuple(batch.src.shape))
    print("visible source tokens:", batch.src_mask[0].tolist())
    print("German input:", batch.tgt[0].tolist())
    print("German labels:", batch.tgt_y[0].tolist())
