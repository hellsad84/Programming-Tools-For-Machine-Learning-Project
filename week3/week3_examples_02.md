# Week 3 validation examples 00005–00009

I checked one detail in each of the five examples using the saved translations in the [Week 2 report](https://github.com/hellsad84/Programming-Tools-For-Machine-Learning-Project/blob/main/docs/week2_results.md#valid-00005).

| ID | Target detail | 1 head: label + evidence | 8 heads: label + evidence |
| --- | --- | --- | --- |
| valid-00005 | The woman jumps. | **Missed.** The line ends with “ein Foto von einem Tisch” (a photo of a table). It leaves out the jump. | **Missed.** The woman’s red top appears in “in einem roten Oberteil”, but nothing says she jumps. |
| valid-00006 | One dog chases another. | **Missed.** “Ein Hund rennt auf dem Boden” describes one dog running on the ground. There is no second dog to chase. | **Missed.** “Ein Hund rennt auf dem Wasser” describes a single dog running on water. |
| valid-00007 | The boy swings a baseball bat. | **Missed.** The boy “steht vor einem Tisch” (stands in front of a table). The bat and the swing are both missing. | **Missed.** The text moves from the boy to “ein kleines Mädchen in einem Gebäude” (a little girl in a building) without mentioning a bat or a swing. |
| valid-00008 | The man is using a telephone. | **Missed.** His only action is “steht auf einem Tisch” (stands on a table). | **Missed.** The line ends with “macht einen großen Gebäude”. There is no mention of a phone or a call. |
| valid-00009 | There is a bicycle. | **Missed.** She still holds something, but “hält einen großen blauen Oberteil” makes it a blue top. | **Kept.** The last word is “Fahrrad” (bicycle). The phrase “in einem Fahrrad” is awkward, but the bicycle is there. |

## Subtotals

| Model | Kept | Missed | Unclear | Total |
| --- | ---: | ---: | ---: | ---: |
| 1 head | 0 | 5 | 0 | 5 |
| 8 heads | 1 | 4 | 0 | 5 |

## Decoding observation

In valid-00005, the eight-head model repeats “die ein Foto” eight times. Just after “die ein Baby”, it uses the phrase three times in a row.
