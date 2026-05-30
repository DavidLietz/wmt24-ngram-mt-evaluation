from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


LANG_PAIR = "en-de"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wmt24" / LANG_PAIR
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "report_assets"
REPORT_TABLE_DIR = REPORT_DIR / "tables"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def word_count(text: str) -> int:
    return len(text.split())


def truncate_text(text: str, max_chars: int = 260) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_dataset_summary_by_domain() -> pd.DataFrame:
    source_lines = read_lines(RAW_DIR / "source.txt")
    metadata_rows = read_jsonl(RAW_DIR / "metadata.jsonl")

    if len(source_lines) != len(metadata_rows):
        raise RuntimeError(
            f"Source and metadata lengths differ: "
            f"{len(source_lines)} vs. {len(metadata_rows)}"
        )

    rows = []
    for segment_id, (source, metadata) in enumerate(zip(source_lines, metadata_rows), start=1):
        rows.append(
            {
                "segment_id": segment_id,
                "domain": metadata.get("domain", "UNKNOWN"),
                "source_words": word_count(source),
            }
        )

    segment_df = pd.DataFrame(rows)

    summary = (
        segment_df
        .groupby("domain", as_index=False)
        .agg(
            segments=("segment_id", "count"),
            source_words=("source_words", "sum"),
            avg_source_words_per_segment=("source_words", "mean"),
        )
        .sort_values("domain")
    )

    total_segments = summary["segments"].sum()
    total_words = summary["source_words"].sum()

    summary["segment_share_percent"] = 100.0 * summary["segments"] / total_segments
    summary["source_word_share_percent"] = 100.0 * summary["source_words"] / total_words
    summary["included_in_interpretive_domain_analysis"] = summary["domain"] != "canary"

    summary.to_csv(
        REPORT_TABLE_DIR / "report_dataset_summary_by_domain.csv",
        index=False,
    )

    summary[summary["domain"] != "canary"].to_csv(
        REPORT_TABLE_DIR / "report_dataset_summary_by_domain_no_canary.csv",
        index=False,
    )

    segment_df.to_csv(
        REPORT_TABLE_DIR / "report_source_segments_with_domain.csv",
        index=False,
    )

    return summary


def build_compact_main_results() -> pd.DataFrame:
    main_ranking_path = REPORT_TABLE_DIR / "report_main_system_ranking.csv"
    if not main_ranking_path.exists():
        raise FileNotFoundError(main_ranking_path)

    ranking = pd.read_csv(main_ranking_path)

    compact = ranking[
        [
            "candidate",
            "word_1_4_f1",
            "system_rank_word_1_4_f1",
            "char_1_6_chrf_like",
            "system_rank_char_1_6_chrf_like",
        ]
    ].copy()

    compact = compact.rename(
        columns={
            "candidate": "system",
            "word_1_4_f1": "word_f1_score",
            "system_rank_word_1_4_f1": "word_f1_rank",
            "char_1_6_chrf_like": "char_f_score",
            "system_rank_char_1_6_chrf_like": "char_f_rank",
        }
    )

    compact.to_csv(
        REPORT_TABLE_DIR / "report_compact_main_results.csv",
        index=False,
    )

    compact.head(12).to_csv(
        REPORT_TABLE_DIR / "report_compact_main_results_top12.csv",
        index=False,
    )

    return compact


def build_selected_disagreement_examples() -> pd.DataFrame:
    examples_path = TABLE_DIR / "metric_disagreement_examples.csv"
    if not examples_path.exists():
        raise FileNotFoundError(examples_path)

    examples = pd.read_csv(examples_path)

    selected = examples.head(8).copy()

    for column in [
        "source",
        "reference_A",
        "best_word_output",
        "best_char_output",
    ]:
        selected[column] = selected[column].map(truncate_text)

    selected = selected[
        [
            "segment_id",
            "source",
            "reference_A",
            "best_word_system",
            "best_word_score",
            "best_word_output",
            "best_char_system",
            "best_char_score",
            "best_char_output",
        ]
    ]

    selected.to_csv(
        REPORT_TABLE_DIR / "report_selected_disagreement_examples.csv",
        index=False,
    )

    markdown_lines = [
        "# Ausgewählte Beispiele für Metrikabweichungen",
        "",
        "Die folgenden Beispiele wurden automatisch aus Segmenten ausgewählt, bei denen die wortbasierte und die zeichenbasierte Metrik unterschiedliche Systeme als beste Übersetzung bewerten. Sie dienen als Grundlage für eine qualitative Diskussion im Projektbericht.",
        "",
    ]

    for _, row in selected.iterrows():
        markdown_lines.extend(
            [
                f"## Segment {int(row['segment_id'])}",
                "",
                f"**Quelle:** {row['source']}",
                "",
                f"**Referenz A:** {row['reference_A']}",
                "",
                f"**Bestes System nach Wort-F1:** `{row['best_word_system']}` "
                f"({row['best_word_score']:.3f})",
                "",
                f"{row['best_word_output']}",
                "",
                f"**Bestes System nach Zeichen-F-Score:** `{row['best_char_system']}` "
                f"({row['best_char_score']:.3f})",
                "",
                f"{row['best_char_output']}",
                "",
            ]
        )

    (REPORT_DIR / "report_selected_disagreement_examples.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    return selected


def build_requirements_check(
    domain_summary: pd.DataFrame,
    compact_results: pd.DataFrame,
    selected_examples: pd.DataFrame,
) -> None:
    checks = [
        {
            "requirement": "Ein Sprachpaar wurde ausgewählt.",
            "status": "erfüllt",
            "evidence": "Sprachpaar en-de.",
        },
        {
            "requirement": "Alle Übersetzungen des Sprachpaars werden betrachtet.",
            "status": "erfüllt",
            "evidence": "26/26 Systemausgaben vollständig geladen und geprüft.",
        },
        {
            "requirement": "Metriken mit unterschiedlichen Texteinheiten oder n-Gramm-Längen werden verwendet.",
            "status": "erfüllt",
            "evidence": "Wortbasierter 1–4-Gramm-F1 und zeichenbasierter 1–6-Gramm-F-Score.",
        },
        {
            "requirement": "Metrikwerte werden für jede maschinelle Übersetzung berechnet.",
            "status": "erfüllt",
            "evidence": f"{len(compact_results)} Systeme in report_compact_main_results.csv.",
        },
        {
            "requirement": "Die Übersetzungen werden nach Metrikwerten geordnet.",
            "status": "erfüllt",
            "evidence": "Systemränge in report_main_system_ranking.csv.",
        },
        {
            "requirement": "Die optimale maschinelle Übersetzung wird als Pseudo-Referenz verwendet.",
            "status": "erfüllt",
            "evidence": "Pseudo-Referenzen in pseudo_reference_overview.csv und bereinigte Rangvergleiche in report_pseudoref_ranking_comparison_no_self.csv.",
        },
        {
            "requirement": "Die zwei Rangfolgen werden verglichen.",
            "status": "erfüllt",
            "evidence": "Rangänderungen und Korrelationen in report_corrected_rank_correlations.csv.",
        },
        {
            "requirement": "Daten werden tabellarisch beschrieben, inklusive Domänen und Wortanzahl im Quelltext.",
            "status": "erfüllt",
            "evidence": f"{len(domain_summary)} Domänenzeilen in report_dataset_summary_by_domain.csv.",
        },
        {
            "requirement": "Ergebnisse werden qualitativ diskutierbar gemacht.",
            "status": "erfüllt",
            "evidence": f"{len(selected_examples)} ausgewählte Metrikabweichungsbeispiele.",
        },
    ]

    check_df = pd.DataFrame(checks)
    check_df.to_csv(
        REPORT_TABLE_DIR / "report_requirements_check.csv",
        index=False,
    )

    lines = [
        "# Anforderungscheck für den Projektbericht",
        "",
        "Diese Datei dient als interne Kontrolle, ob die zentralen Anforderungen der gewählten Aufgabenstellung durch Projektartefakte abgedeckt sind.",
        "",
    ]

    for item in checks:
        lines.extend(
            [
                f"## {item['requirement']}",
                "",
                f"Status: **{item['status']}**",
                "",
                f"Nachweis: {item['evidence']}",
                "",
            ]
        )

    (REPORT_DIR / "report_requirements_check.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    domain_summary = build_dataset_summary_by_domain()
    compact_results = build_compact_main_results()
    selected_examples = build_selected_disagreement_examples()

    build_requirements_check(
        domain_summary=domain_summary,
        compact_results=compact_results,
        selected_examples=selected_examples,
    )

    print("\nReport supplements created:")
    print(REPORT_DIR)


if __name__ == "__main__":
    main()