# Week 3 validation examples 00000–00004

Five validation IDs assigned to Teammate 1 (data owner), one target detail per ID, judged separately for the one-head and eight-head predictions printed in the Week 2 report.

| ID | Target detail | 1 head: label + evidence | 8 heads: label + evidence |
| --- | --- | --- | --- |
| valid-00000 | Loading cotton | Missed: "sitzen auf einem Tisch" (sit on a table); no "lädt" and no "Baumwolle" | Missed: "die auf einem Tisch" (fragment, no verb); no loading action |
| valid-00001 | The man is sleeping | Missed: "spielt" (plays) | Missed: "sitzt" (sits) |
| valid-00002 | The boy has headphones | Missed: "in einem weißen Oberteil" (white top); no "Kopfhörer" | Missed: "in einem blauen Oberteil" (blue top); no "Kopfhörer" |
| valid-00003 | The men are setting up a hut | Missed: "sitzen auf einem Tisch" (sit); no "bauen" and no "Hütte" | Missed: "die auf einem Tisch" (fragment, no verb); no "Hütte" |
| valid-00004 | The man is wearing a life jacket | Missed: "in einem blauen Oberteil" (blue top); no "Rettungsweste" | Missed: "in einem blauen Hemd" (blue shirt); no "Rettungsweste" |

## Subtotals

Each model's three counts add to five.

| Model | Kept | Missed | Unclear | Total |
| --- | --- | --- | --- | --- |
| 1 head | 0 | 5 | 0 | 5 |
| 8 heads | 0 | 5 | 0 | 5 |

## Note on the labels

The eight-head outputs for valid-00000 and valid-00003 stop after a relative clause ("die auf einem Tisch …"): the target action is absent rather than ambiguous, so both rows are recorded as Missed, not Unclear. Every quotation above is copied from the predictions in `docs/week2_results.md`.

## Observation

In valid-00001 the reference verb "schläft" is replaced by "spielt" (plays) with one head and by "sitzt" (sits) with eight heads, so the man survives in the output while the action that carries the meaning disappears.

## Source lines quoted from docs/week2_results.md

### valid-00000 — action: loading cotton

- English source: A group of men are loading cotton onto a truck
- German reference: Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen
- One-head prediction: Eine Gruppe Männer sitzen auf einem Tisch und schauen sich auf einem Gebäude.
- Eight-head prediction: Eine Gruppe von Menschen, die auf einem Tisch, die auf einem Gebäude.

### valid-00001 — action: the man is sleeping (worked example, page 2)

- English source: A man sleeping in a green room on a couch.
- German reference: Ein Mann schläft in einem grünen Raum auf einem Sofa.
- One-head prediction: Ein Mann in einem weißen Oberteil spielt auf einem Tisch.
- Eight-head prediction: Ein Mann in einem blauen Oberteil sitzt auf einem Tisch.

### valid-00002 — object: the boy has headphones

- English source: A boy wearing headphones sits on a woman's shoulders.
- German reference: Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau.
- One-head prediction: Ein Junge in einem weißen Oberteil spielt auf einem Tisch.
- Eight-head prediction: Ein Junge in einem blauen Oberteil sitzt auf einem Tisch.

### valid-00003 — action: the men are setting up a hut

- English source: Two men setting up a blue ice fishing hut on an iced over lake
- German reference: Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf
- One-head prediction: Zwei Männer sitzen auf einem Tisch auf einem Tisch auf einem Tisch auf einem Tisch.
- Eight-head prediction: Zwei Männer, die auf einem Tisch, die auf einem Tisch, während ein Foto.

### valid-00004 — object: the man is wearing a life jacket

- English source: A balding man wearing a red life jacket is sitting in a small boat.
- German reference: Ein Mann mit beginnender Glatze, der eine rote Rettungsweste trägt, sitzt in einem kleinen Boot.
- One-head prediction: Ein Mann in einem blauen Oberteil steht in einem weißen Oberteil in einem blauen Oberteil.
- Eight-head prediction: Ein Mann in einem blauen Hemd spielt in einem blauen Oberteil mit einem blauen Hemd.
