# Berichtsfähige Ergebniszusammenfassung

## Datenstand

- Sprachpaar: en-de
- Bewertete maschinelle Übersetzungssysteme: 26
- Segmente je Quelle/Referenz/System: 998
- Domänen ohne canary: literary, news, social, speech

Die Domäne `canary` wurde aus der inhaltlichen Domänenanalyse ausgeschlossen, weil sie nur ein Segment umfasst und in den aktuellen Ergebnissen für alle Systeme identische Maximalwerte erzeugt.

## Hauptranking gegen menschliche Referenz A

- Bestes System nach wortbasiertem 1–4-Gramm-F1: `TranssionMT` mit 40.191.
- Bestes System nach zeichenbasiertem 1–6-Gramm-F-Score: `Claude-3.5` mit 65.243.
- Die beiden Systemrankings stimmen stark überein: Spearman rho = 0.979, Kendall tau = 0.907.

## Pseudo-Referenz-Experiment

- Für `word_1_4_f1` wurde `TranssionMT` als beste maschinelle Übersetzung gegen Referenz A ausgewählt.
- Für `char_1_6_chrf_like` wurde `Claude-3.5` als beste maschinelle Übersetzung gegen Referenz A ausgewählt.

Die bereinigte Pseudo-Referenz-Auswertung schließt den trivialen Selbstvergleich der jeweils als Referenz genutzten maschinellen Übersetzung aus.

## Stärkste Rangänderungen im Pseudo-Referenz-Experiment

### Wortbasierter 1–4-Gramm-F1
- `ONLINE-G`: Rang 12 gegen Referenz A, Rang 3 gegen Pseudo-Referenz (-9).
- `Dubformer`: Rang 4 gegen Referenz A, Rang 10 gegen Pseudo-Referenz (+6).
- `Claude-3.5`: Rang 2 gegen Referenz A, Rang 5 gegen Pseudo-Referenz (+3).
- `CUNI-NL`: Rang 17 gegen Referenz A, Rang 20 gegen Pseudo-Referenz (+3).
- `Phi-3-Medium`: Rang 19 gegen Referenz A, Rang 16 gegen Pseudo-Referenz (-3).

### Zeichenbasierter 1–6-Gramm-F-Score
- `TranssionMT`: Rang 2 gegen Referenz A, Rang 9 gegen Pseudo-Referenz (+7).
- `ONLINE-B`: Rang 3 gegen Referenz A, Rang 10 gegen Pseudo-Referenz (+7).
- `IOL-Research`: Rang 11 gegen Referenz A, Rang 6 gegen Pseudo-Referenz (-5).
- `Mistral-Large`: Rang 7 gegen Referenz A, Rang 3 gegen Pseudo-Referenz (-4).
- `CommandR-plus`: Rang 8 gegen Referenz A, Rang 4 gegen Pseudo-Referenz (-4).

## Beste Systeme je Domäne

- Domäne `literary`, Zeichenbasierter 1–6-Gramm-F-Score: `TranssionMT` mit 67.369.
- Domäne `literary`, Wortbasierter 1–4-Gramm-F1: `TranssionMT` mit 45.333.
- Domäne `news`, Zeichenbasierter 1–6-Gramm-F-Score: `ONLINE-W` mit 66.492.
- Domäne `news`, Wortbasierter 1–4-Gramm-F1: `ONLINE-W` mit 35.413.
- Domäne `social`, Zeichenbasierter 1–6-Gramm-F-Score: `Dubformer` mit 64.364.
- Domäne `social`, Wortbasierter 1–4-Gramm-F1: `Dubformer` mit 41.204.
- Domäne `speech`, Zeichenbasierter 1–6-Gramm-F-Score: `ONLINE-W` mit 71.364.
- Domäne `speech`, Wortbasierter 1–4-Gramm-F1: `ONLINE-W` mit 43.676.

## Menschliche Referenzen im Pseudo-Referenz-Vergleich

- `reference_B` gegen Pseudo-Referenz `Claude-3.5` bei `char_1_6_chrf_like`: Score 66.528, Rang 17 unter den Nicht-Selbstvergleichen.
- `reference_A` gegen Pseudo-Referenz `Claude-3.5` bei `char_1_6_chrf_like`: Score 64.280, Rang 22 unter den Nicht-Selbstvergleichen.
- `reference_B` gegen Pseudo-Referenz `TranssionMT` bei `word_1_4_f1`: Score 42.901, Rang 18 unter den Nicht-Selbstvergleichen.
- `reference_A` gegen Pseudo-Referenz `TranssionMT` bei `word_1_4_f1`: Score 40.191, Rang 22 unter den Nicht-Selbstvergleichen.