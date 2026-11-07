import torch
from torch.nn.utils.rnn import pad_sequence
from starter_core import Batch

# All 22 English–German training pairs.
PAIRS = [
    ("hello", "hallo"),
    ("i am here", "ich bin hier"),
    ("good night", "gute nacht"),
    ("good morning", "guten morgen"),
    ("good evening", "guten abend"),
    ("thank you", "danke"),
    ("see you soon", "bis bald"),
    ("welcome", "willkommen"),
    ("i am tired", "ich bin müde"),
    ("i am hungry", "ich bin hungrig"),
    ("i am happy", "ich bin glücklich"),
    ("i am ready", "ich bin bereit"),
    ("i need help", "ich brauche hilfe"),
    ("i like music", "ich mag musik"),
    ("i love you", "ich liebe dich"),
    ("the sun shines", "die sonne scheint"),
    ("the water is cold", "das wasser ist kalt"),
    ("the food is good", "das essen ist gut"),
    ("where are you", "wo bist du"),
    ("what is your name", "wie heißt du"),
    ("my name is anna", "ich heiße anna"),
    ("we are learning german", "wir lernen deutsch"),
]

# Reserved token IDs.
PAD, START, END = 0, 1, 2


def make_batch():
    # Build a shared vocabulary for English and German.
    words = sorted({
        word
        for pair in PAIRS
        for text in pair
        for word in text.split()
    })
    vocab = {word: i + 3 for i, word in enumerate(words)}

    # Convert a sentence into token IDs.
    def encode(text):
        ids = [vocab[word] for word in text.split()]
        return torch.tensor(
            [START, *ids, END],
            dtype=torch.long,
        )

    src, tgt = zip(*PAIRS)

    # Pad sentences so every row has the same length.
    def stack(sentences):
        rows = [encode(text) for text in sentences]
        return pad_sequence(
            rows,
            batch_first=True,
            padding_value=PAD,
        )

    # Create the batch before attaching its vocabulary.
    batch = Batch(stack(src), stack(tgt), pad=PAD)
    batch.vocab = vocab

    # Include the three reserved tokens in the vocabulary size.
    return batch, len(vocab) + 3

if __name__ == "__main__":
    batch, size = make_batch()

    print("Number of sentence pairs:", len(PAIRS))
    print("Vocabulary size:", size)
    print("Source shape:", tuple(batch.src.shape))
    print("Visible source tokens:", batch.src_mask[0].tolist())
    print("German input token IDs:", batch.tgt[0].tolist())
    print("German label token IDs:", batch.tgt_y[0].tolist())

    print("\nAll training pairs:")
    for number, (english, german) in enumerate(PAIRS, start=1):
        print(f"{number:02d}. English: {english} | German: {german}")

    print("\nData preparation completed successfully.")
