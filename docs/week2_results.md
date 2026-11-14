# Week 2 CPU pilot results

Owner: Teammate 3 / project admin. Recorded on 14 November 2026.

## Outcome

We completed a matched one-head versus eight-head English-to-German comparison on a small Multi30k dataset. Both models trained for 500 updates and were evaluated on all 1,014 validation sentences. Eight heads scored 6.40 BLEU versus 5.63 for one head, a difference of 0.77 BLEU points in this run. Translation quality remains limited, and this single-seed pilot does not establish a general winner or reproduce the paper's WMT14 results.

| Heads | Updates | Validation NLL | BLEU | Training seconds | Decoding seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 4.1199 | 5.63 | 10.70 | 3.84 |
| 8 | 500 | 4.0628 | 6.40 | 12.38 | 14.59 |

BLEU is better when higher; validation NLL is better when lower. BLEU is not a percentage of correctly translated sentences.

## Environment and source version

- Machine: user's MacBook Pro, ARM64; exact CPU model was not recorded in the run files.
- Recorded platform: `macOS-15.4.1-arm64-arm-64bit-Mach-O`.
- Python: 3.13.15; PyTorch: 2.15.0; CPU execution with four threads.
- Installed dependencies: [requirements-week2.txt](../requirements-week2.txt). This is the installed environment snapshot, not a guarantee of compatibility with every platform.
- Source commit inspected when preparing this report: `1ff6ff11025bff58c3a55567c1ca0564049aba56`. The training scripts do not embed a Git commit in their output; this is a separately recorded source reference. No tracked differences were found in the model, trainer, evaluator, configuration, or tests during inspection.
- Each run's environment and resolved settings are copied into [results/week2](../results/week2/).

## Data and integrity

- Multi30k Task 1, English to German; upstream revision `2cc0f2d7c615b60e07e2d51bb5edc004444d7e0e`.
- Prepared training / validation / test pairs: 5,000 / 1,014 / 1,000.
- One shared training-only SentencePiece BPE tokenizer; 4,000 vocabulary entries.
- Training subset seed: 11. Maximum training length in the delivered configuration: 100 tokens, including BOS/EOS.
- Data preparation and selection are described in [week2_data.md](week2_data.md).
- The reserved test split was not used for model scoring in these experiments.
- Manifest SHA-256: `b2d8da359415e5c094fb2ea7a96e63ef572d3550fca133bbe601f5908074e270`.
- Tokenizer SHA-256: `2afa8fab7d32560607f267792125e0391680b13d0eb4f6440341db57fd08e1ad`.

The frozen dataset's manifest checks passed. The reproducibility test compares text with normalized line endings to accommodate Windows CRLF and macOS LF; the exact-byte artifact integrity checks remain in place.

## Matched experiment settings

| Setting | Value for both runs |
| --- | --- |
| Encoder / decoder layers | 1 / 1 |
| Model width / feed-forward width | 128 / 256 |
| Dropout | 0.1 |
| Parameters | 1,871,776 |
| Training / evaluation batch size | 16 / 16 |
| Updates / seed | 500 / 11 |
| Optimizer | Adam, learning rate 0.001, betas (0.9, 0.999), eps 1e-8 |
| Weight decay / gradient clipping | 0.0 / 1.0 |
| Device / CPU threads | CPU / 4 |
| Decoder | Greedy, at most 128 new tokens |
| Training examples processed | 7,992, including repeated examples |
| Non-padding training target tokens | 138,671 |

Only head count changes between the saved configurations. With 5,000 training pairs, 7,992 processed examples is approximately 1.60 passes through the training split. The final partial batch is retained, so the total is slightly below 500 × 16.

Both runs have the same batch-order digest:
`2d22ebd72b842bbf270d6ee7673b61445bc107b0e2ff34c1e956a3b24972b1cc`.

## Checks and training behavior

The full setup check and all 21 Week 2 tests passed before the measured experiments. The paired-result check passed. Both final evaluation outputs contain 1,014 aligned sources, references, IDs, and predictions, with no empty prediction lines. BLEU was independently recomputed from the saved prediction/reference files when preparing this report and matched both saved scores.

The first logged training interval (step 50) had NLL 6.6418 for one head and 6.6314 for eight heads. The final logged interval (step 500) had NLL 4.0989 and 4.0365 respectively. These are interval token-weighted training losses, not full-training-set losses. The reduction shows learning during training, while the validation scores and examples show that translation quality is still weak.

## Timing trials and timing interpretation

Before the measured runs, both models completed 20 updates and evaluated the first 50 validation sentences. Their trial BLEU scores were 0.1312 and 0.1571. These subset measurements were used only as workflow checks, not as the final comparison. No advance full-runtime estimate was recorded; actual measured run timings are reported above.

Training time covers batch retrieval and training updates. It excludes model/environment setup, checkpoint writes, and evaluation. Decoding time covers translation generation, not all validation loss computation or process startup. These values are not total terminal-command runtimes. Eight-head decoding took longer in this run; output lengths and timing variability can affect it, so this is not a general speed claim.

SacreBLEU signature:
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`

## Ten validation examples

These are the first ten validation IDs in file order, selected without choosing favorable outputs. Each source and reference is matched to both models' predictions.

### valid-00000

- **English source:** A group of men are loading cotton onto a truck
- **German reference:** Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen
- **One-head prediction:** Eine Gruppe Männer sitzen auf einem Tisch und schauen sich auf einem Gebäude.
- **Eight-head prediction:** Eine Gruppe von Menschen, die auf einem Tisch, die auf einem Gebäude.

### valid-00001

- **English source:** A man sleeping in a green room on a couch.
- **German reference:** Ein Mann schläft in einem grünen Raum auf einem Sofa.
- **One-head prediction:** Ein Mann in einem weißen Oberteil spielt auf einem Tisch.
- **Eight-head prediction:** Ein Mann in einem blauen Oberteil sitzt auf einem Tisch.

### valid-00002

- **English source:** A boy wearing headphones sits on a woman's shoulders.
- **German reference:** Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau.
- **One-head prediction:** Ein Junge in einem weißen Oberteil spielt auf einem Tisch.
- **Eight-head prediction:** Ein Junge in einem blauen Oberteil sitzt auf einem Tisch.

### valid-00003

- **English source:** Two men setting up a blue ice fishing hut on an iced over lake
- **German reference:** Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf
- **One-head prediction:** Zwei Männer sitzen auf einem Tisch auf einem Tisch auf einem Tisch auf einem Tisch.
- **Eight-head prediction:** Zwei Männer, die auf einem Tisch, die auf einem Tisch, während ein Foto.

### valid-00004

- **English source:** A balding man wearing a red life jacket is sitting in a small boat.
- **German reference:** Ein Mann mit beginnender Glatze, der eine rote Rettungsweste trägt, sitzt in einem kleinen Boot.
- **One-head prediction:** Ein Mann in einem blauen Oberteil steht in einem weißen Oberteil in einem blauen Oberteil.
- **Eight-head prediction:** Ein Mann in einem blauen Hemd spielt in einem blauen Oberteil mit einem blauen Hemd.

### valid-00005

- **English source:** A lady in a red coat, holding a bluish hand bag likely of asian descent, jumping off the ground for a snapshot.
- **German reference:** Eine Frau in einem rotem Mantel, die eine vermutlich aus Asien stammende Handtasche in einem blauen Farbton hält, springt für einen Schnappschuss in die Luft.
- **One-head prediction:** Eine Frau in einem weißen Oberteil und einem blauen T-Shirt und ein Foto von einem Tisch.
- **Eight-head prediction:** Eine Frau in einem roten Oberteil, die ein Baby, die ein Foto, die ein Foto, die ein Foto, während ein Foto, während ein Mann in der andere Frau, während ein Foto von einer Person, während ein Foto von einem Mann, während ein Mann in der Nähe von einem Bull von einer Person, während ein Foto von einer Person, während ein Foto, die ein Foto von einer Person, während ein Foto von einer Person, die andere Frau, die ein Foto von einem Krift, die ein Foto von einer Person, die ein Foto, die ein Foto von einer Person, während ein Foto, die in der ein Foto von

### valid-00006

- **English source:** A brown dog is running after the black dog.
- **German reference:** Ein brauner Hund rennt dem schwarzen Hund hinterher.
- **One-head prediction:** Ein Hund rennt auf dem Boden und hält einen Ball.
- **Eight-head prediction:** Ein Hund rennt auf dem Wasser.

### valid-00007

- **English source:** A young boy wearing a Giants jersey swings a baseball bat at an incoming pitch.
- **German reference:** Ein kleiner Junge mit einem Giants-Trikot schwingt einen Baseballschläger in Richtung eines ankommenden Balls.
- **One-head prediction:** Ein Junge in einem weißen Oberteil steht in einem weißen T-Shirt in einem blauen T-Shirt steht vor einem Tisch.
- **Eight-head prediction:** Ein Junge in einem roten Oberteil, die ein Foto, während ein kleines Mädchen in einem Tisch in einem Tisch, während ein kleines Mädchen in einem Gebäude.

### valid-00008

- **English source:** A man in a cluttered office is using the telephone
- **German reference:** Ein Mann telefoniert in einem unaufgeräumten Büro
- **One-head prediction:** Ein Mann in einem weißen Oberteil steht auf einem Tisch.
- **Eight-head prediction:** Ein Mann in einem blauen Oberteil macht einen großen Gebäude.

### valid-00009

- **English source:** A smiling woman in a peach tank top stands holding a mountain bike
- **German reference:** Eine lächelnde Frau mit einem pfirsichfarbenen Trägershirt hält ein Mountainbike
- **One-head prediction:** Eine Frau in einem weißen Oberteil hält einen großen blauen Oberteil mit einem Tisch mit einem Tisch.
- **Eight-head prediction:** Eine Frau in einem roten Oberteil sitzt auf einem blauen Oberteil, während ein Mann in einem Fahrrad.

## Interpretation and limitations

Eight heads achieved higher BLEU and lower validation NLL in this experiment. However, examples show missing objects, substituted actions, and repeated generic sentence patterns. For example, the sentence about a man sleeping on a couch is translated as a man playing or sitting on a table. Producing German-looking text does not mean the source meaning is preserved.

The experiment uses one seed, 5,000 training pairs, a small one-layer model, and only 500 updates. No uncertainty estimate or significance test was performed. The results do not justify a general claim that eight heads always outperform one head. They also should not be compared directly with the paper's reported WMT14 results.

Later weeks can add multiple seeds, a larger matched training budget, and peer review. Week 2 did not require peer-review sign-off. Keep the test split reserved while developing the experiment.

## Reproduction

Run from the repository root. Place the entire original frozen package under `data/week2/`, including the manifest and tokenizer. Do not create a manifest manually or retrain the tokenizer for just one run. If an older checkout has the dataset directly under `data/`, copy the complete package into the expected subfolder first; in that checkout, a blank file named `data/week2` also needs to be renamed before creating the folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-week2.txt
python -m week2.check_setup --full
python -m week2.data_pipeline --check --data-dir data/week2
python -m week2.model --check
python -m unittest discover -s tests -p 'test_week2*.py' -v
python -m week2.train --heads 1 --seed 11 --out runs/week2/h1_s11
python -m week2.train --heads 8 --seed 11 --out runs/week2/h8_s11
python -m week2.evaluate --checkpoint runs/week2/h1_s11/final.pt --out runs/week2/h1_s11/eval
python -m week2.evaluate --checkpoint runs/week2/h8_s11/final.pt --out runs/week2/h8_s11/eval
python -m week2.summarize
```

These commands are for fresh output directories. The saved runs already exist on the experiment machine. To repeat them, use new run/evaluation paths and pass the corresponding paths with `--one`, `--eight`, and `--out` to `week2.summarize`.

## Saved artifacts

Small evidence files included in GitHub:

- [Comparison summary](../results/week2/summary.md).
- [One-head metrics](../results/week2/h1_s11/metrics.json), [training log](../results/week2/h1_s11/train_log.csv), [configuration](../results/week2/h1_s11/resolved_config.json), and [environment](../results/week2/h1_s11/environment.json).
- [Eight-head metrics](../results/week2/h8_s11/metrics.json), [training log](../results/week2/h8_s11/train_log.csv), [configuration](../results/week2/h8_s11/resolved_config.json), and [environment](../results/week2/h8_s11/environment.json).

Complete local experiment records are stored under the admin's project clone at `~/Programming-Tools-For-Machine-Learning-Project/`:

- `runs/week2/h1_s11/` and `runs/week2/h8_s11/`: checkpoints, configurations, environment records and training logs.
- Each run's `eval/`: full prediction, reference, source and ID files plus metrics.
- `runs/timing/`: short trial outputs.
- `data/week2/`: the frozen dataset and tokenizer.

Checkpoints, the complete dataset, and full run directories are excluded from this upload package. No shared-storage upload has been verified; the full artifacts currently remain in the local clone. The small evidence files are copies, so the original run outputs are preserved.
