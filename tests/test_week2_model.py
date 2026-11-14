"""Owner: Teammate 2. Uses fake tensors, not Teammate 1's data."""

import unittest
import torch

from starter_core import Batch
from week2.model import build_model, FIXTURE_CONFIG
from week2.decode import translate_batch


class FakeTokenizer:
    def bos_id(self):
        return 1

    def eos_id(self):
        return 2

    def pad_id(self):
        return 0

    def decode(self, ids):
        words = {
            4: "token4",
            5: "token5",
            6: "token6",
            7: "token7",
        }
        return " ".join(words[i] for i in ids)





class FakeGenerator:
    def __init__(self, schedules):
        self.schedules = schedules

    def __call__(self, hidden):
        batch_size = hidden.size(0)

        scores = torch.full(
            (batch_size, 8),
            -10.0,
            device=hidden.device
        )

        for row in range(batch_size):
            step = int(hidden[row, 0].item())

            if step >= len(self.schedules[row]):
                token = 2
            else:
                token = self.schedules[row][step]

            scores[row, token] = 10.0

        return scores




class FakeModel:
    def __init__(self, schedules):
        self.generator = FakeGenerator(schedules)

    def eval(self):
        return self

    def encode(self, src, src_mask):
        return src

    def decode(self, memory, src_mask, tgt, tgt_mask):
        batch_size = tgt.size(0)

        hidden = torch.zeros(
            batch_size,
            tgt.size(1),
            2,
            device=tgt.device
        )

        hidden[:, -1, 0] = tgt.size(1) - 1

        hidden[:, -1, 1] = torch.arange(
            batch_size,
            device=tgt.device
        )

        return hidden


class ModelTests(unittest.TestCase):

    def test_causal_mask(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

        model = build_model(
            16, 8, FIXTURE_CONFIG
        ).eval()

        src = torch.tensor([[1, 4, 2]])
        full_tgt = torch.tensor([[1, 5, 6, 2]])

        batch = Batch(src, full_tgt, pad=0)

        changed = batch.tgt.clone()
        changed[0, 2] = 7

        with torch.no_grad():
            a = model(
                src, batch.tgt,
                batch.src_mask, batch.tgt_mask
            )

            b = model(
                src, changed,
                batch.src_mask, batch.tgt_mask
            )

        self.assertTrue(
            torch.allclose(
                a[:, :2], b[:, :2], atol=1e-6
            )
        )

    def test_equal_parameters_and_initial_weights(self):
        torch.manual_seed(11)
        model_1 = build_model(16, 1, FIXTURE_CONFIG)

        torch.manual_seed(11)
        model_8 = build_model(16, 8, FIXTURE_CONFIG)

        params_1 = sum(
            p.numel() for p in model_1.parameters()
        )

        params_8 = sum(
            p.numel() for p in model_8.parameters()
        )

        self.assertEqual(params_1, params_8)

        state_1 = model_1.state_dict()
        state_8 = model_8.state_dict()

        self.assertEqual(
            list(state_1.keys()),
            list(state_8.keys())
        )

        for key in state_1:
            self.assertTrue(
                torch.equal(state_1[key], state_8[key]),
                msg=f"Initial weights differ: {key}"
            )

    def test_immediate_eos(self):
        model = FakeModel([
            [2],
            [2],
        ])

        tokenizer = FakeTokenizer()

        src = torch.tensor([
            [3, 4],
            [3, 5],
        ])

        outputs = translate_batch(
            model,
            src,
            None,
            tokenizer
        )

        self.assertEqual(outputs, ["", ""])

    def test_token_then_eos(self):
        model = FakeModel([
            [4, 2],
        ])

        tokenizer = FakeTokenizer()

        src = torch.tensor([[3, 4]])

        outputs = translate_batch(
            model,
            src,
            None,
            tokenizer
        )

        self.assertEqual(outputs, ["token4"])

    def test_output_limit(self):
        model = FakeModel([
            [4, 5, 6],
        ])

        tokenizer = FakeTokenizer()

        src = torch.tensor([[3, 4]])

        outputs = translate_batch(
            model,
            src,
            None,
            tokenizer,
            max_new_tokens=3
        )

        self.assertEqual(
            outputs,
            ["token4 token5 token6"]
        )

    def test_batch_rows_finish_at_different_steps(self):
        model = FakeModel([
            [4, 2],
            [5, 6, 2],
        ])

        tokenizer = FakeTokenizer()

        src = torch.tensor([
            [3, 4],
            [3, 5],
        ])

        outputs = translate_batch(
            model,
            src,
            None,
            tokenizer
        )

        self.assertEqual(
            outputs,
            ["token4", "token5 token6"]
        )


if __name__ == "__main__":
    unittest.main()
