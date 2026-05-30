# Vergleich n-Gramm-basierter Metriken zur MT-Evaluation

Dieses Repository enthält eine Python-Pipeline zur automatischen Bewertung maschineller Übersetzungen mit n-Gramm-basierten Textähnlichkeitsmetriken.

## Projektziel

Untersucht wird, wie stabil Systemrankings maschineller Übersetzung ausfallen, wenn eine wortbasierte und eine zeichenbasierte n-Gramm-Metrik verwendet werden. Zusätzlich wird analysiert, wie sich die Rankings verändern, wenn die jeweils beste maschinelle Übersetzung als Pseudo-Referenz eingesetzt wird.

## Datengrundlage

Verwendet wird der öffentliche WMT24-News-Systems-Datensatz für das Sprachpaar Englisch → Deutsch (`en-de`).

Die Pipeline lädt folgende Daten automatisch aus dem öffentlichen Repository:

- Quelltexte
- menschliche Referenzen
- Metadaten
- maschinelle Systemausgaben

Die Rohdaten werden nicht versioniert, sondern bei Bedarf durch `src/download_wmt24.py` neu geladen.

## Metriken

Die Pipeline berechnet zwei selbst implementierte n-Gramm-basierte Metriken:

1. `word_1_4_f1`  
   Wortbasierter F1-Score über 1- bis 4-Gramme.

2. `char_1_6_chrf_like`  
   Zeichenbasierter F-Score über 1- bis 6-Gramme mit stärkerer Recall-Gewichtung.

Die Ergebnisse werden segmentweise berechnet und anschließend systemweise gemittelt.

## Ausführung

Empfohlen wird Python 3.11 oder neuer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/run_pipeline.py
```

## Projektstruktur

```txt
src/
  download_wmt24.py
  inspect_wmt24.py
  evaluate_metrics.py
  build_report_assets.py
  build_report_supplements.py
  run_pipeline.py

outputs/
  report_assets/
    tables/
    figures/
```

## Zentrale Ausgaben

Nach erfolgreicher Ausführung befinden sich berichtsnahe Tabellen und Abbildungen unter:

```txt
outputs/report_assets/
```

Wichtige Ergebnisdateien sind unter anderem:

```txt
outputs/report_assets/report_key_findings.md
outputs/report_assets/report_requirements_check.md
outputs/report_assets/tables/report_main_system_ranking.csv
outputs/report_assets/tables/report_dataset_summary_by_domain.csv
outputs/report_assets/tables/report_corrected_rank_correlations.csv
outputs/report_assets/tables/report_pseudoref_ranking_comparison_no_self.csv
outputs/report_assets/figures/
```

## Reproduzierbarkeit

Die Pipeline lädt die Daten neu, prüft die Vollständigkeit der Systemausgaben, berechnet alle Metriken und erzeugt die berichtsnahen Ergebnisartefakte reproduzierbar aus den Rohdaten.
