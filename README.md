# Attention Is All You Need — Week 1 Progress

**Status: Week 1 practice completed. The replication project is still in progress.**

This README records what we completed in Week 1. We will update it with new setup instructions, experiments and results as the project develops.

## Project purpose

Our project is based on [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762). We plan to compare one attention head with eight attention heads in a small English-to-German Transformer, inspired by Table 3(A).

For Week 1, our goal was to set up the repository and get a small practice example working from data preparation to German prediction output.

## What we completed in Week 1

- Set up the GitHub repository and divided the main coding tasks between three teammates.
- Added 22 hand-written English–German sentence pairs.
- Prepared the vocabulary, token IDs, padding and attention masks.
- Built and checked models with one and eight attention heads.
- Checked that each model can complete a training update.
- Added a demo that trains for 200 updates and prints predictions for all 22 pairs.
- Ran the practice successfully through GitHub Actions.

**Latest successful workflow runtime: 45 seconds**, as recorded by the team. This is the workflow runtime, not the time for one sentence or one model. Future runs may take a different amount of time.

This is a small training exercise. We have not yet used WMT14, evaluated unseen sentences or reproduced the paper's BLEU results. A successful workflow also does not automatically mean all 22 translations are correct.

## Files and team responsibilities

This table lists the files added or adapted for the current practice.

| File | Week 1 purpose or change | Responsibility |
| --- | --- | --- |
| `data_pipeline.py` | Added the 22 pairs, vocabulary, padded batches and data checks | Teammate 1 |
| `model.py` | Added a small model wrapper and checks for one/eight heads | Teammate 2 |
| `train.py` | Added a one-update training check and loss output | Teammate 3 |
| `translation_demo.py` | Added repeated training and readable German predictions | Shared integration |
| `starter_core.py` | Extracted selected Transformer and batch definitions from the credited upstream code | Shared dependency |
| `.github/workflows/week1-check.yml` | Adapted a workflow template to install dependencies and run our scripts | Shared workflow |
| `README.md` | Documents Week 1 progress and instructions | Documentation |
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

The workflow uses **Python 3.13, PyTorch 2.5.1 (CPU), and NumPy 2.1.3** on an Ubuntu runner. It installs the dependencies with:

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

## What the output shows

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

Outputs are available in the Actions logs. The current scripts do not save trained models.

## Original code and license

We use selected code from [Harvard NLP's Annotated Transformer](https://github.com/harvardnlp/annotated-transformer), credited to Alexander Rush and the upstream contributors.

- **Starting commit:** [`debc9fd747bb2123160a98046ad1c2d4da44a567`](https://github.com/harvardnlp/annotated-transformer/commit/debc9fd747bb2123160a98046ad1c2d4da44a567).
- **Original source file:** `the_annotated_transformer.py`.
- **Reused code:** selected model and batch definitions in `starter_core.py`, with imports limited to the practice dependencies.
- **License:** MIT; the copyright and permission notice is included in [LICENSE-upstream.txt](LICENSE-upstream.txt).

The reused Transformer implementation is credited to its original authors. Our practice scripts and file changes are listed above.

## Next steps

- Choose and document a suitable real translation dataset or subset.
- Prepare separate training and evaluation data.
- Plan a fair one-head versus eight-head comparison with the same data and training budget.
- Add measured results and updated instructions to this README as the work progresses.

