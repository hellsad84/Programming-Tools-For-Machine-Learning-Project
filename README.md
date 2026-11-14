# Attention Is All You Need — Week 1 and Week 2 Progress

**Status: Week 1 practice and the Week 2 CPU pilot completed. The replication project is still in progress.**

This README records what we completed in Weeks 1 and 2, including setup instructions and measured experiment results. We will continue updating it as the project develops.

## Project purpose

Our project is based on [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762). We compare one attention head with eight attention heads in a small English-to-German Transformer, inspired by Table 3(A).

For Week 1, our goal was to set up the repository and get a small practice example working from data preparation to German prediction output.

For Week 2, our goal was to use a real translation dataset, prepare separate training and evaluation data, and run a fair comparison with the same data and training budget.

## What we completed in Week 1

- Set up the GitHub repository and divided the main coding tasks between three teammates.
- Added 22 hand-written English–German sentence pairs.
- Prepared the vocabulary, token IDs, padding and attention masks.
- Built and checked models with one and eight attention heads.
- Checked that each model can complete a training update.
- Added a demo that trains for 200 updates and prints predictions for all 22 pairs.
- Ran the practice successfully through GitHub Actions.

**Latest successful Week 1 workflow runtime: 45 seconds**, as recorded by the team. This is the workflow runtime, not the time for one sentence or one model. Future runs may take a different amount of time.

At the end of Week 1, this was a small training exercise. We had not yet used WMT14, evaluated unseen sentences or reproduced the paper's BLEU results. A successful workflow also does not automatically mean all 22 translations are correct.

## Week 1 files and team responsibilities

This table lists the files added or adapted for the Week 1 practice.

| File | Week 1 purpose or change | Responsibility |
| --- | --- | --- |
| `data_pipeline.py` | Added the 22 pairs, vocabulary, padded batches and data checks | 黄天恩 |
| `model.py` | Added a small model wrapper and checks for one/eight heads | 田锐 |
| `train.py` | Added a one-update training check and loss output | 赵晟凯 |
| `translation_demo.py` | Added repeated training and readable German predictions | Shared integration |
| `starter_core.py` | Extracted selected Transformer and batch definitions from the credited upstream code | Shared dependency |
| `.github/workflows/week1-check.yml` | Adapted a workflow template to install dependencies and run our scripts | Shared workflow |
| `README.md` | Documents progress and instructions | Documentation |
| `docs/starting_code.md` | Records the upstream source, commit and license | Documentation |
| `docs/starting-code.md` | Additional file currently containing a model-code example | Documentation |
| `.gitignore` | Excludes datasets, checkpoints and `.env` | Repository setup |
| `LICENSE-upstream.txt` | Preserves the original MIT license notice | Attribution |

The source record is `docs/starting_code.md` with an underscore; the similarly named file with a hyphen is separate.

## How to run the Week 1 practice

GitHub Actions sets up the environment automatically. No local installation or dataset download is needed for this method.

1. Open this repository and click **Actions**.
2. Select **Week 1 practice check**.
3. Click **Run workflow**, select `main`, then confirm **Run workflow**.
4. Open the new run and click the **practice** job.
5. Expand the steps to read their outputs.

After editing the code, start a new workflow run so it uses the updated files.

The Week 1 workflow uses **Python 3.13, PyTorch 2.5.1 (CPU), and NumPy 2.1.3** on an Ubuntu runner. It installs the dependencies with:

```bash
python -m pip install numpy==2.1.3
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

It then runs these commands from the repository's main folder:

```bash
python data_pipeline.py
python model.py
python train.py 1 11
python train.py 8 11
python translation_demo.py 1
python translation_demo.py 8
```

For `train.py`, the arguments are the head count and random seed. For `translation_demo.py`, the argument is the head count; the script uses seed 11 and 200 training updates. Each command starts a fresh model.

## What the Week 1 output shows

| Step | Expected output |
| --- | --- |
| Data preparation | 22 pairs, vocabulary size 83, source shape `(22, 6)`, and the reference pairs |
| Model checks | One-head and eight-head models both produce shape `(22, 5, 128)` |
| One-update training | Environment versions, loss and confirmation through checks that weights changed |
| Translation demos | Training losses, all 22 English sentences with German references and predictions, and an exact-match count |

To see the generated translations, expand **Show German predictions with one attention head** or **Show German predictions with eight attention heads** in the workflow logs.

Each prediction is printed alongside its English input and German reference. The final line has this format:

```text
Training-set exact matches: X/22
```

`X` is the number of predictions that exactly match the reference. These are the same sentences used for training, so this count checks learning of the practice examples. It does not measure translation of new sentences. The exact counts have not yet been recorded in this README.

Week 1 outputs are available in the Actions logs. The Week 1 scripts do not save trained models.

## What we completed in Week 2

- Used a real English–German translation dataset: Multi30k Task 1.
- Prepared 5,000 training pairs, 1,014 validation pairs and 1,000 reserved test pairs.
- Used one shared SentencePiece BPE tokenizer with a vocabulary of 4,000.
- Implemented data loading, model construction, greedy decoding, training and evaluation.
- Passed all 21 Week 2 tests before the measured experiments.
- Completed short trials for both head counts.
- Trained both models for 500 updates using the same data and settings.
- Evaluated both models on all 1,014 validation sentences.
- Saved checkpoints, training logs, predictions and measured results.

The test split remains reserved. These experiments use validation results.

### Week 2 responsibilities

| Teammate | Work completed |
| --- | --- |
| Teammate 1 | Data preparation, tokenizer, batching, data tests and data report |
| Teammate 2 | Model construction, decoding, model tests and model report |
| Teammate 3 / admin | Integration checks, training, evaluation, comparison summary, results report and README |

Each teammate owned their component. Peer review is planned for later weeks.

## Week 2 experiment settings

Both models used the same configuration except for the number of attention heads.

| Setting | Value |
| --- | --- |
| Translation direction | English to German |
| Dataset | Multi30k Task 1 |
| Training pairs | 5,000 |
| Validation pairs | 1,014 |
| Reserved test pairs | 1,000 |
| Tokenizer | Shared SentencePiece BPE |
| Vocabulary size | 4,000 |
| Encoder / decoder layers | 1 / 1 |
| Model width | 128 |
| Feed-forward width | 256 |
| Dropout | 0.1 |
| Attention heads | 1 or 8 |
| Parameters per model | 1,871,776 |
| Training updates | 500 |
| Batch size | 16 |
| Random seed | 11 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Device | CPU |
| CPU threads | 4 |
| Decoding | Greedy, at most 128 new tokens |

The recorded Week 2 environment used Python 3.13.15 and PyTorch 2.15.0 on macOS ARM64. The installed package versions are recorded in `requirements-week2.txt`.

Both runs processed 7,992 training examples, including repeated examples, and 138,671 non-padding target tokens. Data fingerprints, batch order and the remaining configuration settings matched.

## Week 2 measured results

| Heads | Updates | Validation NLL | BLEU | Training seconds | Decoding seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 4.1199 | 5.63 | 10.70 | 3.84 |
| 8 | 500 | 4.0628 | 6.40 | 12.38 | 14.59 |

Both evaluations used all **1,014 validation sentences**.

- Higher BLEU is better.
- Lower validation NLL is better.
- Eight heads scored **0.77 BLEU points higher** in this run.
- Eight heads also achieved slightly lower validation loss.

Training time covers batch retrieval and training updates. It excludes setup, checkpoint saving and evaluation. Decoding time covers translation generation, not the entire evaluation command. These values are not total terminal-command runtimes.

SacreBLEU signature:

```text
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
```

### Interpretation and limitations

Both models learned during training, but translation quality remains weak. The inspected predictions sometimes change actions or objects and repeat generic sentence patterns.

For example, the source sentence about a man sleeping in a green room on a couch was translated into sentences about a man playing or sitting on a table. Producing German-looking text does not mean the source meaning is preserved.

The eight-head model performed better on the aggregate metrics in this experiment. However, this comparison uses only one seed, a small dataset, a small model and 500 updates. It does not establish that eight heads always perform better.

This is a CPU pilot inspired by the paper. It does not reproduce the paper's WMT14 experiment or BLEU results.

### Detailed reports

- [Data preparation report](docs/week2_data.md)
- [Model and decoder report](docs/week2_model.md)
- [Full results, ten translation examples and limitations](docs/week2_results.md)
- [Measured comparison summary](results/week2/summary.md)

## How to run the Week 2 experiment

Run the commands from the repository root, where `starter_core.py`, `configs/` and `week2/` are located.

### 1. Prepare the environment

On macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-week2.txt
```

For later terminal sessions, activate the existing environment:

```bash
source .venv/bin/activate
```

### 2. Place the prepared dataset

Place the complete frozen data package under:

```text
data/week2/
```

The folder should contain:

```text
data/week2/
├── manifest.json
├── audit.json
├── train.jsonl
├── valid.jsonl
├── test.jsonl
├── tokenizer.model
├── tokenizer.vocab
└── tokenizer_input.txt
```

Keep every file listed in the manifest. Both head counts must use the same dataset and tokenizer.

The source code on GitHub alone is not the prepared dataset. If the delivered files are directly under `data/`, place the complete package inside the `data/week2/` subfolder.

### 3. Run the checks

```bash
python -m week2.check_setup --full
python -m week2.data_pipeline --check --data-dir data/week2
python -m week2.model --check
python -m unittest discover -s tests -p 'test_week2*.py' -v
```

The test suite should finish with:

```text
Ran 21 tests ...
OK
```

### 4. Run short trials

Train each model for 20 updates and evaluate the first 50 validation sentences:

```bash
python -m week2.train --heads 1 --steps 20 --out runs/timing/h1
python -m week2.evaluate --checkpoint runs/timing/h1/final.pt --limit 50 --out runs/timing/h1/eval

python -m week2.train --heads 8 --steps 20 --out runs/timing/h8
python -m week2.evaluate --checkpoint runs/timing/h8/final.pt --limit 50 --out runs/timing/h8/eval
```

These are short workflow and timing checks. They are not the final comparison.

### 5. Run the measured experiments

Confirm that `configs/week2_cpu.json` uses 500 updates. Run the models sequentially on the same computer:

```bash
python -m week2.train --heads 1 --seed 11 --out runs/week2/h1_s11
python -m week2.train --heads 8 --seed 11 --out runs/week2/h8_s11
```

Keep the dataset and configuration unchanged between runs.

### 6. Evaluate the full validation set

```bash
python -m week2.evaluate --checkpoint runs/week2/h1_s11/final.pt --out runs/week2/h1_s11/eval
python -m week2.evaluate --checkpoint runs/week2/h8_s11/final.pt --out runs/week2/h8_s11/eval
```

Do not use `--limit` for these evaluations. Each model should evaluate all 1,014 validation sentences.

### 7. Generate the comparison

```bash
python -m week2.summarize
```

This checks that the paired runs are comparable and writes:

```text
results/week2/summary.md
```

Use fresh output directories for repeat runs. If a run or summary already exists, choose new paths and supply the corresponding `--one`, `--eight` and `--out` paths to the summary command.

## Where Week 2 outputs are saved

Each full run saves its outputs locally:

```text
runs/week2/
├── h1_s11/
│   ├── resolved_config.json
│   ├── environment.json
│   ├── train_log.csv
│   ├── final.pt
│   └── eval/
│       ├── metrics.json
│       ├── valid.src.en
│       ├── valid.ref.de
│       ├── valid.pred.de
│       └── valid.ids.txt
└── h8_s11/
    ├── resolved_config.json
    ├── environment.json
    ├── train_log.csv
    ├── final.pt
    └── eval/
        ├── metrics.json
        ├── valid.src.en
        ├── valid.ref.de
        ├── valid.pred.de
        └── valid.ids.txt
```

Additional intermediate checkpoints may also be present.

- `final.pt`: trained model checkpoint.
- `train_log.csv`: training progress and recorded timing.
- `metrics.json`: BLEU, validation loss and experiment details.
- `valid.src.en`: English inputs.
- `valid.ref.de`: reference German translations.
- `valid.pred.de`: generated German translations.
- `valid.ids.txt`: sentence IDs used to align the outputs.

Small result summaries and copies of supporting logs are stored under `results/week2/`. Keep the complete dataset, Python environment and model checkpoints out of ordinary Git commits.

The prepared `.gitignore` excludes `data/`, `runs/`, `.venv/`, `__pycache__/` and `*.pt`. Ignore rules do not remove files that Git already tracks.

## Original code and license

We use selected code from [Harvard NLP's Annotated Transformer](https://github.com/harvardnlp/annotated-transformer), credited to Alexander Rush and the upstream contributors.

- **Starting commit:** [`debc9fd747bb2123160a98046ad1c2d4da44a567`](https://github.com/harvardnlp/annotated-transformer/commit/debc9fd747bb2123160a98046ad1c2d4da44a567).
- **Original source file:** `the_annotated_transformer.py`.
- **Reused code:** selected model and batch definitions in `starter_core.py`, with imports limited to the practice dependencies.
- **License:** MIT; the copyright and permission notice is included in [LICENSE-upstream.txt](LICENSE-upstream.txt).

The reused Transformer implementation is credited to its original authors. Our practice scripts and file changes are listed above.

## Next steps

- Keep the Week 2 results as the measured baseline.
- Back up the complete experiment artifacts to team storage and record their shared location.
- In later weeks, repeat the comparison with multiple seeds and a larger matched training budget.
- Examine translation errors and carry out the planned peer review.
- Keep the test split reserved during development.
- Continue adding measured results and updated instructions as the work progresses.

