"""Owner: Teammate 2. Guide section 14. Never pass German references here."""
import torch
from starter_core import subsequent_mask



@torch.no_grad()
def translate_batch(
    model,
    src,
    src_mask,
    tokenizer,
    max_new_tokens=128
):
    if max_new_tokens < 1:
        raise ValueError(
            "max_new_tokens must be positive"
        )

    model.eval()

    bos = tokenizer.bos_id()
    eos = tokenizer.eos_id()
    pad = tokenizer.pad_id()

    memory = model.encode(src, src_mask)

    generated = torch.full(
        (src.size(0), 1),
        bos,
        dtype=torch.long,
        device=src.device
    )

    finished = torch.zeros(
        src.size(0),
        dtype=torch.bool,
        device=src.device
    )

    for _ in range(max_new_tokens):

        mask = subsequent_mask(
            generated.size(1)
        ).to(src.device)

        hidden = model.decode(
            memory,
            src_mask,
            generated,
            mask
        )

        scores = model.generator(hidden[:, -1])

        scores[:, pad] = float("-inf")
        scores[:, bos] = float("-inf")

        next_id = scores.argmax(dim=-1)

        next_id = torch.where(
            finished,
            torch.full_like(next_id, eos),
            next_id
        )

        generated = torch.cat(
            [generated, next_id[:, None]],
            dim=1
        )

        finished = finished | (next_id == eos)

        if bool(finished.all()):
            break

    outputs = []

    for row in generated.tolist():

        ids = []

        for token in row[1:]:

            if token == eos:
                break

            if token not in (pad, bos):
                ids.append(token)

        outputs.append(tokenizer.decode(ids))

    return outputs
