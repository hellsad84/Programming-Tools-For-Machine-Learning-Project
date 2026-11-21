# Week 3 progress

We reviewed the first ten saved English-to-German validation examples. For each sentence, we asked whether each model preserved one specified detail. The one-head model missed all ten details. The eight-head model preserved one, the bicycle in `valid-00009`.

## Results

| Model | Kept | Missed | Unclear | Details reviewed |
| --- | ---: | ---: | ---: | ---: |
| 1 head | 0 | 10 | 0 | 10 |
| 8 heads | 1 | 9 | 0 | 10 |

Teammate 1 reviewed IDs 00000–00004 and recorded five misses for each model. Teammate 2 reviewed IDs 00005–00009 and recorded five misses for one head, and one kept detail and four misses for eight heads. Teammate 3 checked the ten IDs against the saved predictions and combined the counts. [Teammate 1’s annotations](week3_examples_01.md) · [Teammate 2’s annotations](week3_examples_02.md)

## What the translations show

- Both models sometimes preserved the subject while changing the action. In `valid-00001`, both retained the man, but the source says he is sleeping. The one-head prediction says “spielt” (plays) and the eight-head prediction says “sitzt” (sits). In `valid-00006`, both retained a running dog but omitted the second dog it was chasing. [Saved translations](../docs/week2_results.md#ten-validation-examples)
- The only difference in labels occurred at `valid-00009`. The eight-head prediction contains “Fahrrad” (bicycle). The one-head prediction says the woman holds a blue top. “Kept” applies only to the presence of the bicycle. The eight-head translation still changes who is with the bicycle and what they are doing. [Bicycle example](../docs/week2_results.md#valid-00009)
- In `valid-00005`, the eight-head prediction repeats “die ein Foto” eight times, including three consecutive occurrences. Both models omit the woman’s jump. [Jumping example](../docs/week2_results.md#valid-00005)

## Week 2 baseline

Both models used seed 11, trained for 500 updates, and were evaluated on the same 1,014 validation sentences. Eight heads scored 0.77 BLEU points higher in this run. [Week 2 comparison](../results/week2/summary.md)

| Attention heads | Validation BLEU | Validation NLL | Saved metrics |
| --- | ---: | ---: | --- |
| 1 | 5.63 | 4.1199 | [One-head metrics](../results/week2/h1_s11/metrics.json) |
| 8 | 6.40 | 4.0628 | [Eight-head metrics](../results/week2/h8_s11/metrics.json) |

## Limits and next step

The manual counts cover one detail in each of the first ten validation sentences from one seed. They cannot establish how often either model preserves meaning across the full validation set. A follow-up would be to select a fresh random sample of validation IDs, choose each target detail from the source before reading either prediction, and repeat the paired review. The test split remains reserved.

Both local run folders contain the final checkpoint and validation outputs. The shared backup location is still unverified.
