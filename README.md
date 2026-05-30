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