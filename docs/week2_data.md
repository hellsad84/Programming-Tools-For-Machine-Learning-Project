# Week 2 data preparation (Teammate 1)

Data pipeline for the English–German Transformer: download Multi30k, filter the training
split, train one shared BPE tokenizer, and freeze `data/week2/` together with a manifest.
All numbers below come from the run in section "Commands".

## Source data

- Multi30k Task 1 raw files (`data/task1/raw/`), revision `2cc0f2d7c615b60e07e2d51bb5edc004444d7e0e`
  pinned in `configs/week2_data.json`
- Six files: `train.{en,de}.gz`, `val.{en,de}.gz`, `test_2016_flickr.{en,de}.gz`
- Not used: the `tok/` tokenized release, French/Czech files, image features, Task 2 captions
- Upstream: https://github.com/multi30k/dataset (Multi30k, Elliott et al. 2016; extended by
  Barrault et al. 2018). The project code licence does not cover the dataset; upstream terms apply.

Line counts checked during download (English and German agree per split):
train 29,000, valid 1,014, test 1,000.

## Selection rule

Exact matching only, no near-duplicate detection:

1. NFC-normalize and trim every sentence before comparing.
2. Drop a training row whose normalized source appears in validation or test
   (`source_overlap_with_held_out`).
3. Drop a training row whose normalized `(src, tgt)` pair is already kept, keeping the lowest id
   (`duplicate_training_pair`).
4. Shuffle the remaining rows with `subset_seed` and take the first `subset_size` (max 5,000).
5. Length-filter the sample with `max_train_tokens`, where a row's length is
   `len(encode(text)) + 2` (BOS and EOS).

Sampling comes after deduplication, so the subset size does not change when duplicates are removed
later. Held-out splits are not rewritten; validation/test source overlap is only recorded.

## Counts

| Quantity | Value |
|---|---|
| Upstream training rows per language | 29,000 |
| Upstream validation / test rows per language | 1,014 / 1,000 |
| Removed training rows | 4 |
| — `source_overlap_with_held_out` | 1 (`train-04749`) |
| — `duplicate_training_pair` | 3 (`train-14214`, `train-16866`, `train-20148`) |
| Eligible training rows | 28,996 |
| Removed by the length filter | 0 |
| Final training rows | 5,000 |
| Validation / test rows | 1,014 / 1,000 |
| `valid ∩ test` shared sources | 0 |

## Tokenizer

- SentencePiece BPE trained jointly on English and German
- Corpus: `tokenizer_input.txt`, 10,000 lines (5,000 training pairs × 2 languages); no
  validation or test sentence is added
- Target vocabulary 4,000; measured `get_piece_size()` = 4,000
- Special ids `pad=0, bos=1, eos=2, unk=3`, asserted in `train_tokenizer`
- `character_coverage=1.0`, `normalization_rule_name="identity"`, `hard_vocab_limit=False`
- Normalization is done by our own NFC helper, so raw German references stay untouched for scoring
- One tokenizer is reused by both head configurations and is never retrained per run

## Artifacts

`manifest.json` stores the config, the download receipt (URL, sha256 and line count per source
file), the split counts, the vocabulary size, the special ids, and the hash of every artifact below.

| File | sha256 |
|---|---|
| `train.jsonl` | `fe0bab93ed011fbc0f83e41282d9b210deab7d582b5d3c84871775b84360705f` |
| `valid.jsonl` | `7b287e59544a7665fd18bbe22ed850f3420fa21212f34bfd34c0ba1dc89f18de` |
| `test.jsonl` | `2b4027b0bff78527362a3d4fbd74a3c366314f9ed4b7adc14eb8be32f670f186` |
| `tokenizer.model` | `2afa8fab7d32560607f267792125e0391680b13d0eb4f6440341db57fd08e1ad` |
| `tokenizer.vocab` | `3b1f6f1ec65db21fd16367ceec52ca6120f32cc87328e058922ace04bbb62ed6` |
| `audit.json` | `820d5b6dabd7ff3bd339558bbfe6f06f363c7ed98490305c603f732bc32f02db` |

`tokenizer_input.txt` is written as well (10,000 lines) but is not hashed in the manifest; it is
only inspection material.

`audit.json` keeps the removed rows with reasons, the eligible count, the sampled candidate ids,
the final training ids, the length-filtered ids, and the validation/test overlap count.

## Batches

- Week 1 `Batch` class reused: the full German sequence keeps BOS and EOS, so
  `[BOS, a, b, EOS, PAD]` gives decoder input `[BOS, a, b, EOS]` and labels `[a, b, EOS, PAD]`
- `pad=0` is passed explicitly instead of relying on the class default
- For batch size B, source length S and target length T: `.src [B,S]`, `.tgt [B,T-1]`,
  `.tgt_y [B,T-1]`, `.src_mask [B,1,S]`, `.tgt_mask [B,T-1,T-1]`; `.ntokens` counts non-PAD labels
- `.ids`, `.source_texts`, `.references` keep the raw rows for scoring
- The loader uses its own seeded generator with `num_workers=0`; training shuffles, validation keeps
  a fixed order, and the generator is independent of the model's generator

## Commands

```text
python -m week2.prepare_data --config configs/week2_data.json --download-only
  -> Downloaded and verified: 6 files

python -m week2.prepare_data --config configs/week2_data.json
  -> Prepared: {'train': 5000, 'valid': 1014, 'test': 1000} vocab: 4000

python -m week2.data_pipeline --check --data-dir data/week2
  -> counts: {'train': 5000, 'valid': 1014, 'test': 1000}
     vocab_size: 4000
     special_ids: {'pad': 0, 'bos': 1, 'eos': 2, 'unk': 3}
     src (2, 20) tgt (2, 19) tgt_y (2, 19)
     src_mask (2, 1, 20) tgt_mask (2, 19, 19)
     ids: ['train-15678', 'train-13213']
     ntokens: 31

python -m unittest discover -s tests -p test_week2_data.py
  -> Ran 9 tests ... OK
```

## Tests

`tests/test_week2_data.py` holds 9 tests: four on selection, three on batching, two on the frozen
package.

1. Duplicate rows and held-out sources are removed from fixture rows, with the expected reasons.
2. A fixed `subset_seed` always produces the same subset.
3. Selected sources do not intersect held-out sources (fixture rows).
4. The same intersection is empty on the real package.
5. Padding sits only on the right and is 0; `src_mask` hides exactly the PAD keys.
6. Labels are one token ahead of the decoder input; PAD is excluded from `.ntokens`.
7. The target mask never allows attending to a later key.
8. A second `prepare` into another directory reproduces the counts, the vocabulary size, the final
   ids and byte-identical `train/valid/test.jsonl`. `tokenizer.model` is not byte-identical across
   runs because the binary embeds its model path, which is why the frozen tokenizer is distributed.
9. A copy of one artifact with a flipped byte is rejected by `load_artifacts`.

## Notes

- Deduplication is exact-only; near-duplicate detection is not implemented.
- The 5,000-pair cap makes this a small CPU pilot, not a reproduction of the WMT14 numbers.
- `max_train_tokens` is 100 and removed 0 rows here, so the setting is not binding for this corpus.
- Validation and test share no normalized source, so no overlap needs to be reported as a risk.
