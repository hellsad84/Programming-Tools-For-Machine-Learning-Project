# Week 2 measured validation results

| Heads | Updates | Validation NLL | BLEU | Train seconds | Decode seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | 4.1199 | 5.63 | 10.70 | 3.84 |
| 8 | 500 | 4.0628 | 6.40 | 12.38 | 14.59 |

BLEU difference (8 minus 1): 0.77
Evaluated pairs: 1014
SacreBLEU signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`

One CPU pilot with one seed; this does not establish a general winner.

Environment, source-version notes, examples and artifact locations: [Week 2 report](../../docs/week2_results.md).

