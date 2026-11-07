import sys

import torch
import torch.nn.functional as F

from data_pipeline import PAD, START, END, PAIRS, make_batch
from model import build_model
from starter_core import subsequent_mask


def run(heads=8, steps=200):
    torch.manual_seed(11)
    torch.set_num_threads(2)

    batch, vocab_size = make_batch()
    id_to_word = {
        number: word for word, number in batch.vocab.items()
    }

    model = build_model(vocab_size, heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Attention heads: {heads}")
    print(f"Training examples: {len(PAIRS)}")
    print(f"Training updates: {steps}")
    print("Practice demo: predictions on the training sentences.")
    print("This is not a test of translation on unseen sentences.\n")

    # Train repeatedly so the model can learn the small sentence list.
    model.train()

    for step in range(1, steps + 1):
        hidden = model(
            batch.src,
            batch.tgt,
            batch.src_mask,
            batch.tgt_mask,
        )
        scores = model.generator(hidden)

        loss = F.nll_loss(
            scores.reshape(-1, vocab_size),
            batch.tgt_y.reshape(-1),
            ignore_index=PAD,
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is not finite.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0:
            print(f"Update {step}: loss = {loss.item():.4f}")

    # Generate German tokens using only each English input.
    model.eval()
    exact_matches = 0

    with torch.no_grad():
        for index, (english, reference) in enumerate(PAIRS):
            source = batch.src[index:index + 1]
            source_mask = batch.src_mask[index:index + 1]
            memory = model.encode(source, source_mask)

            generated = torch.tensor([[START]], dtype=torch.long)
            predicted_words = []
            ended = False

            for _ in range(20):
                target_mask = subsequent_mask(generated.size(1))
                hidden = model.decode(
                    memory, source_mask, generated, target_mask
                )
                scores = model.generator(hidden[:, -1])

                # Padding and start markers are not output words.
                scores[:, PAD] = float("-inf")
                scores[:, START] = float("-inf")
                next_id = scores.argmax(dim=-1).item()

                if next_id == END:
                    ended = True
                    break

                predicted_words.append(id_to_word[next_id])
                generated = torch.cat(
                    [generated, torch.tensor([[next_id]])],
                    dim=1,
                )

            prediction = " ".join(predicted_words)
            if ended and prediction == reference:
                exact_matches += 1

            print(f"\nEnglish:          {english}")
            print(f"German reference: {reference}")
            print(f"German prediction: {prediction or '(empty)'}")

            if not ended:
                print("Stopped at the 20-token generation limit.")

    print(
        f"\nTraining-set exact matches: "
        f"{exact_matches}/{len(PAIRS)}"
    )


if __name__ == "__main__":
    heads = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run(heads=heads)
