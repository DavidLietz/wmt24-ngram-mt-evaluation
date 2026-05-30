from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "report_assets"
REPORT_TABLE_DIR = REPORT_DIR / "tables"
REPORT_FIGURE_DIR = REPORT_DIR / "figures"

METRICS = ["word_1_4_f1", "char_1_6_chrf_like"]

METRIC_LABELS = {
    "word_1_4_f1": "Wortbasierter 1–4-Gramm-F1",
    "char_1_6_chrf_like": "Zeichenbasierter 1–6-Gramm-F-Score",
}


def rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(method="min", ascending=False).astype(int)


def read_csv(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)


def write_main_system_ranking(metric_scores: pd.DataFrame) -> pd.DataFrame:
    systems = metric_scores[metric_scores["candidate_type"] == "system"].copy()

    for metric in METRICS:
        systems[f"system_rank_{metric}"] = rank_desc(systems[metric])

    systems["rank_difference_char_minus_word"] = (
        systems["system_rank_char_1_6_chrf_like"]
        - systems["system_rank_word_1_4_f1"]
    )

    systems = systems[
        [
            "candidate",
            "word_1_4_f1",
            "system_rank_word_1_4_f1",
            "char_1_6_chrf_like",
            "system_rank_char_1_6_chrf_like",
            "rank_difference_char_minus_word",
        ]
    ].sort_values(["system_rank_word_1_4_f1", "candidate"])

    systems.to_csv(REPORT_TABLE_DIR / "report_main_system_ranking.csv", index=False)
    return systems


def write_pseudo_reference_comparison(
    main_system_ranking: pd.DataFrame,
    pseudo_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for metric in METRICS:
        subset = pseudo_scores[
            (pseudo_scores["pseudo_reference_metric"] == metric)
            & (pseudo_scores["candidate_type"] == "system")
            & (~pseudo_scores["is_pseudo_reference_self"])
        ].copy()

        initial = main_system_ranking[["candidate", metric]].copy()
        initial = initial[initial["candidate"].isin(subset["candidate"])]
        initial = initial.rename(columns={metric: "score_human_reference"})
        initial["rank_human_reference"] = rank_desc(initial["score_human_reference"])

        pseudo = subset[
            ["candidate", "pseudo_reference", "score"]
        ].rename(columns={"score": "score_pseudo_reference"})
        pseudo["rank_pseudo_reference"] = rank_desc(pseudo["score_pseudo_reference"])

        merged = initial.merge(pseudo, on="candidate", how="inner")
        merged["metric"] = metric
        merged["metric_label"] = METRIC_LABELS[metric]
        merged["rank_delta"] = (
            merged["rank_pseudo_reference"] - merged["rank_human_reference"]
        )

        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)
    result = result[
        [
            "metric",
            "metric_label",
            "pseudo_reference",
            "candidate",
            "score_human_reference",
            "rank_human_reference",
            "score_pseudo_reference",
            "rank_pseudo_reference",
            "rank_delta",
        ]
    ].sort_values(["metric", "rank_human_reference", "candidate"])

    result.to_csv(REPORT_TABLE_DIR / "report_pseudoref_ranking_comparison_no_self.csv", index=False)
    return result


def write_human_reference_pseudoref_scores(pseudo_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric in METRICS:
        subset = pseudo_scores[
            (pseudo_scores["pseudo_reference_metric"] == metric)
            & (~pseudo_scores["is_pseudo_reference_self"])
        ].copy()

        subset["rank_among_non_self_candidates"] = rank_desc(subset["score"])

        human = subset[subset["candidate_type"] == "human_reference"].copy()
        rows.append(human)

    result = pd.concat(rows, ignore_index=True)
    result = result[
        [
            "pseudo_reference_metric",
            "pseudo_reference",
            "candidate",
            "score",
            "rank_among_non_self_candidates",
        ]
    ].sort_values(["pseudo_reference_metric", "rank_among_non_self_candidates"])

    result.to_csv(REPORT_TABLE_DIR / "report_human_reference_scores_against_pseudoref.csv", index=False)
    return result


def write_domain_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    domain_summary = read_csv("domain_summary.csv")
    domain_scores = read_csv("domain_metric_scores_reference_A.csv")

    domain_summary_no_canary = domain_summary[domain_summary["domain"] != "canary"].copy()
    total_segments = domain_summary_no_canary["segments"].sum()
    domain_summary_no_canary["share_percent_without_canary"] = (
        100.0 * domain_summary_no_canary["segments"] / total_segments
    )

    domain_summary_no_canary.to_csv(
        REPORT_TABLE_DIR / "report_domain_summary_no_canary.csv",
        index=False,
    )

    domain_system_scores = domain_scores[
        (domain_scores["domain"] != "canary")
        & (domain_scores["candidate_type"] == "system")
    ].copy()

    ranked_parts = []
    for metric in METRICS:
        part = domain_system_scores.copy()
        part["metric"] = metric
        part["metric_label"] = METRIC_LABELS[metric]
        part["score"] = part[metric]
        part["domain_rank"] = (
            part.groupby("domain")["score"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        ranked_parts.append(
            part[
                [
                    "domain",
                    "metric",
                    "metric_label",
                    "candidate",
                    "score",
                    "domain_rank",
                ]
            ]
        )

    domain_rankings = pd.concat(ranked_parts, ignore_index=True)
    domain_rankings = domain_rankings.sort_values(["domain", "metric", "domain_rank", "candidate"])

    domain_rankings.to_csv(
        REPORT_TABLE_DIR / "report_domain_system_rankings_no_canary.csv",
        index=False,
    )

    domain_top5 = domain_rankings[domain_rankings["domain_rank"] <= 5].copy()
    domain_top5.to_csv(
        REPORT_TABLE_DIR / "report_domain_top5_systems_no_canary.csv",
        index=False,
    )

    return domain_summary_no_canary, domain_rankings


def write_corrected_correlations(
    main_system_ranking: pd.DataFrame,
    pseudo_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    spearman = spearmanr(
        main_system_ranking["system_rank_word_1_4_f1"],
        main_system_ranking["system_rank_char_1_6_chrf_like"],
    )
    kendall = kendalltau(
        main_system_ranking["system_rank_word_1_4_f1"],
        main_system_ranking["system_rank_char_1_6_chrf_like"],
    )

    rows.append(
        {
            "comparison": "Systemrankings: Wort-1–4-F1 vs. Zeichen-1–6-F-Score",
            "scope": "26 systems, human reference A, system-only ranks",
            "spearman_rho": float(spearman.statistic),
            "spearman_pvalue": float(spearman.pvalue),
            "kendall_tau": float(kendall.statistic),
            "kendall_pvalue": float(kendall.pvalue),
        }
    )

    for metric in METRICS:
        subset = pseudo_comparison[pseudo_comparison["metric"] == metric].copy()

        spearman = spearmanr(
            subset["rank_human_reference"],
            subset["rank_pseudo_reference"],
        )
        kendall = kendalltau(
            subset["rank_human_reference"],
            subset["rank_pseudo_reference"],
        )

        rows.append(
            {
                "comparison": f"{METRIC_LABELS[metric]}: Human reference vs. pseudo-reference",
                "scope": "25 systems, pseudo-reference self-comparison excluded",
                "spearman_rho": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_pvalue": float(kendall.pvalue),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(REPORT_TABLE_DIR / "report_corrected_rank_correlations.csv", index=False)
    return result


def plot_main_top_systems(main_system_ranking: pd.DataFrame) -> None:
    for metric in METRICS:
        rank_col = f"system_rank_{metric}"
        top = main_system_ranking.sort_values(rank_col).head(12)

        fig, ax = plt.subplots(figsize=(12, 7.5))
        ax.barh(top["candidate"], top[metric])
        ax.invert_yaxis()
        ax.set_xlabel("Durchschnittlicher Segmentwert", fontsize=12)
        ax.set_ylabel("System", fontsize=12)
        ax.set_title(f"Top-12-Systeme nach {METRIC_LABELS[metric]}", fontsize=13, pad=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
        fig.tight_layout(pad=1.2)
        fig.savefig(REPORT_FIGURE_DIR / f"report_top12_{metric}.png", dpi=300)
        plt.close(fig)


def plot_pseudoref_rank_delta(pseudo_comparison: pd.DataFrame) -> None:
    for metric in METRICS:
        subset = pseudo_comparison[pseudo_comparison["metric"] == metric].copy()
        subset = subset.sort_values("rank_delta")

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.barh(subset["candidate"], subset["rank_delta"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Rangänderung (Pseudo-Referenz minus menschliche Referenz)", fontsize=12)
        ax.set_ylabel("System", fontsize=12)
        ax.set_title(f"Rangänderungen ohne Selbstvergleich: {METRIC_LABELS[metric]}", fontsize=13, pad=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
        fig.tight_layout(pad=1.2)
        fig.savefig(REPORT_FIGURE_DIR / f"report_pseudoref_rank_delta_{metric}.png", dpi=300)
        plt.close(fig)


def write_key_findings(
    main_system_ranking: pd.DataFrame,
    pseudo_reference_overview: pd.DataFrame,
    pseudo_comparison: pd.DataFrame,
    human_pseudoref_scores: pd.DataFrame,
    domain_summary_no_canary: pd.DataFrame,
    domain_rankings: pd.DataFrame,
    corrected_correlations: pd.DataFrame,
) -> None:
    top_word = main_system_ranking.sort_values("system_rank_word_1_4_f1").iloc[0]
    top_char = main_system_ranking.sort_values("system_rank_char_1_6_chrf_like").iloc[0]

    corr_main = corrected_correlations.iloc[0]

    strongest_changes = (
        pseudo_comparison
        .assign(abs_delta=lambda df: df["rank_delta"].abs())
        .sort_values(["metric", "abs_delta"], ascending=[True, False])
        .groupby("metric")
        .head(5)
    )

    domain_top = (
        domain_rankings[domain_rankings["domain_rank"] == 1]
        .sort_values(["domain", "metric"])
    )

    lines = [
        "# Berichtsfähige Ergebniszusammenfassung",
        "",
        "## Datenstand",
        "",
        "- Sprachpaar: en-de",
        "- Bewertete maschinelle Übersetzungssysteme: 26",
        "- Segmente je Quelle/Referenz/System: 998",
        f"- Domänen ohne canary: {', '.join(domain_summary_no_canary['domain'].astype(str))}",
        "",
        "Die Domäne `canary` wurde aus der inhaltlichen Domänenanalyse ausgeschlossen, weil sie nur ein Segment umfasst und in den aktuellen Ergebnissen für alle Systeme identische Maximalwerte erzeugt.",
        "",
        "## Hauptranking gegen menschliche Referenz A",
        "",
        f"- Bestes System nach wortbasiertem 1–4-Gramm-F1: `{top_word['candidate']}` mit {top_word['word_1_4_f1']:.3f}.",
        f"- Bestes System nach zeichenbasiertem 1–6-Gramm-F-Score: `{top_char['candidate']}` mit {top_char['char_1_6_chrf_like']:.3f}.",
        f"- Die beiden Systemrankings stimmen stark überein: Spearman rho = {corr_main['spearman_rho']:.3f}, Kendall tau = {corr_main['kendall_tau']:.3f}.",
        "",
        "## Pseudo-Referenz-Experiment",
        "",
    ]

    for _, row in pseudo_reference_overview.iterrows():
        lines.append(
            f"- Für `{row['metric']}` wurde `{row['pseudo_reference']}` als beste maschinelle Übersetzung gegen Referenz A ausgewählt."
        )

    lines.extend(
        [
            "",
            "Die bereinigte Pseudo-Referenz-Auswertung schließt den trivialen Selbstvergleich der jeweils als Referenz genutzten maschinellen Übersetzung aus.",
            "",
            "## Stärkste Rangänderungen im Pseudo-Referenz-Experiment",
            "",
        ]
    )

    for metric in METRICS:
        label = METRIC_LABELS[metric]
        lines.append(f"### {label}")
        metric_changes = strongest_changes[strongest_changes["metric"] == metric]
        for _, row in metric_changes.iterrows():
            lines.append(
                f"- `{row['candidate']}`: Rang {int(row['rank_human_reference'])} gegen Referenz A, "
                f"Rang {int(row['rank_pseudo_reference'])} gegen Pseudo-Referenz "
                f"({int(row['rank_delta']):+d})."
            )
        lines.append("")

    lines.extend(
        [
            "## Beste Systeme je Domäne",
            "",
        ]
    )

    for _, row in domain_top.iterrows():
        lines.append(
            f"- Domäne `{row['domain']}`, {row['metric_label']}: `{row['candidate']}` mit {row['score']:.3f}."
        )

    lines.extend(
        [
            "",
            "## Menschliche Referenzen im Pseudo-Referenz-Vergleich",
            "",
        ]
    )

    for _, row in human_pseudoref_scores.iterrows():
        lines.append(
            f"- `{row['candidate']}` gegen Pseudo-Referenz `{row['pseudo_reference']}` "
            f"bei `{row['pseudo_reference_metric']}`: Score {row['score']:.3f}, "
            f"Rang {int(row['rank_among_non_self_candidates'])} unter den Nicht-Selbstvergleichen."
        )

    (REPORT_DIR / "report_key_findings.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metric_scores = read_csv("metric_scores_reference_A.csv")
    pseudo_scores = read_csv("metric_scores_pseudoref.csv")
    pseudo_reference_overview = read_csv("pseudo_reference_overview.csv")

    main_system_ranking = write_main_system_ranking(metric_scores)
    pseudo_comparison = write_pseudo_reference_comparison(main_system_ranking, pseudo_scores)
    human_pseudoref_scores = write_human_reference_pseudoref_scores(pseudo_scores)
    domain_summary_no_canary, domain_rankings = write_domain_assets()
    corrected_correlations = write_corrected_correlations(main_system_ranking, pseudo_comparison)

    plot_main_top_systems(main_system_ranking)
    plot_pseudoref_rank_delta(pseudo_comparison)

    write_key_findings(
        main_system_ranking=main_system_ranking,
        pseudo_reference_overview=pseudo_reference_overview,
        pseudo_comparison=pseudo_comparison,
        human_pseudoref_scores=human_pseudoref_scores,
        domain_summary_no_canary=domain_summary_no_canary,
        domain_rankings=domain_rankings,
        corrected_correlations=corrected_correlations,
    )

    print("\nReport assets created:")
    print(REPORT_DIR)


if __name__ == "__main__":
    main()