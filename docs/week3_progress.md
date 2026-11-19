# Week 3 progress

The baseline and sample checks are complete. The combined translation findings are pending delivery of the two annotation files.

## Week 2 baseline

Both models used seed 11 and ran for 500 updates. Each was evaluated on the same 1,014 validation sentences. The saved metrics in the Week 2 upload package match the original evaluation records.

| Attention heads | Validation BLEU | Validation NLL | Evidence |
| --- | ---: | ---: | --- |
| 1 | 5.63 | 4.1199 | [One-head metrics](../results/week2/h1_s11/metrics.json) |
| 8 | 6.40 | 4.0628 | [Eight-head metrics](../results/week2/h8_s11/metrics.json) |

Eight heads scored 0.77 BLEU points higher in this single-seed pilot. [Week 2 comparison](../results/week2/summary.md)

## Ten validation examples

The review uses the first ten validation IDs in saved order. Both run folders contain 1,014 unique IDs, and their IDs, sources and references match in order. For each of the first ten IDs, the source, reference and both predictions printed in the [Week 2 report](week2_results.md) match the original evaluation files.

| Owner | Assigned IDs | Expected contribution |
| --- | --- | --- |
| Teammate 1 | valid-00000 through valid-00004 | Five paired judgments in week3_examples_01.md |
| Teammate 2 | valid-00005 through valid-00009 | Five paired judgments and a decoding observation in week3_examples_02.md |
| Teammate 3 | All ten IDs | Baseline verification, coverage checks and combined findings |

## Findings

Each reviewer checks one specified detail per sentence using the labels Kept, Missed and Unclear. Each model will have ten judgments once both files are included.

| Model | Kept | Missed | Unclear | Reviewed |
| --- | ---: | ---: | ---: | ---: |
| 1 head | Pending | Pending | Pending | Pending |
| 8 heads | Pending | Pending | Pending | Pending |

## Artifact status and next step

Both local run folders contain a nonempty final checkpoint and validation outputs. The shared backup location is unverified.

After delivery, link each annotation file above, check one row from each against the saved predictions, then add the two subtotals for each label. Confirm each model has ten judgments. Replace this instruction with one finding supported by two IDs, a limitation of this ten-sentence sample, and one follow-up. Add the Week 3 report link to the root README.
