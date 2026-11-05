# Programming-Tools-For-Machine-Learning-Project

## Attention Is All You Need: replication plan 
We will test whether eight attention heads improve English-to-German
translation compared with one head in a reduced-scale Transformer.
## Published comparison
Table 3(A), newstest2013 development set (higher BLEU is better):
- One head: 24.9 BLEU.
- Eight heads (base): 25.8 BLEU. Difference: +0.9 BLEU.
## Our first milestone
Train on a small WMT14 English-German subset and decode one example.
For the head comparison, fix the data, model width and training budget.
