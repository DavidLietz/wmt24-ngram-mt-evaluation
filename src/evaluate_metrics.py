from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau, spearmanr


LANG_PAIR = "en-de"
MAIN_REFERENCE = "reference_A"
ALT_REFERENCE = "reference_B"
MAIN_METRICS = ["word_1_4_f1", "char_1_6_chrf_like"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wmt24" / LANG_PAIR
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class Candidate:
    name: str
    candidate_type: str
    lines: list[str]


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


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def normalize_chars(text: str) -> str:
    return " ".join(text.lower().split())


def word_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def char_ngrams(text: str, n: int) -> Counter[str]:
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def overlap_counts(hyp_ngrams: Counter, ref_ngrams: Counter) -> int:
    return sum((hyp_ngrams & ref_ngrams).values())


def precision_recall_fscore_from_counters(
    hypothesis_units,
    reference_units,
    n_values: Iterable[int],
    beta: float = 1.0,
    counter_fn=word_ngrams,
) -> float:
    scores: list[float] = []

    for n in n_values:
        hyp_ngrams = counter_fn(hypothesis_units, n)
        ref_ngrams = counter_fn(reference_units, n)

        if not hyp_ngrams and not ref_ngrams:
            continue
        if not hyp_ngrams or not ref_ngrams:
            scores.append(0.0)
            continue

        overlap = overlap_counts(hyp_ngrams, ref_ngrams)
        precision = overlap / sum(hyp_ngrams.values())
        recall = overlap / sum(ref_ngrams.values())

        if precision == 0.0 and recall == 0.0:
            scores.append(0.0)
            continue

        beta_sq = beta * beta
        f_score = (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)
        scores.append(f_score)

    if not scores:
        return 0.0
    return 100.0 * sum(scores) / len(scores)


def word_1_4_f1(hypothesis: str, reference: str) -> float:
    return precision_recall_fscore_from_counters(
        word_tokens(hypothesis),
        word_tokens(reference),
        [1, 2, 3, 4],
        beta=1.0,
        counter_fn=word_ngrams,
    )


def char_1_6_chrf_like(hypothesis: str, reference: str) -> float:
    # chrF-inspired: character n-gram F-score with beta=2.
    # This weights recall higher, which is common in chrF-style MT evaluation.
    return precision_recall_fscore_from_counters(
        normalize_chars(hypothesis),
        normalize_chars(reference),
        [1, 2, 3, 4, 5, 6],
        beta=2.0,
        counter_fn=char_ngrams,
    )


METRICS: dict[str, Callable[[str, str], float]] = {
    "word_1_4_f1": word_1_4_f1,
    "char_1_6_chrf_like": char_1_6_chrf_like,
}


def validate_equal_lengths(name_to_lines: dict[str, list[str]]) -> None:
    lengths = {name: len(lines) for name, lines in name_to_lines.items()}
    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) != 1:
        raise RuntimeError(f"Line counts differ: {lengths}")


def load_candidates(reference_b_lines: list[str]) -> list[Candidate]:
    system_dir = RAW_DIR / "system_outputs"
    candidates = [
        Candidate(
            name=ALT_REFERENCE,
            candidate_type="human_reference",
            lines=reference_b_lines,
        )
    ]

    for path in sorted(system_dir.glob("*.txt")):
        candidates.append(
            Candidate(
                name=path.stem,
                candidate_type="system",
                lines=read_lines(path),
            )
        )

    return candidates


def build_segment_scores(candidates: list[Candidate], reference_lines: list[str]) -> pd.DataFrame:
    rows = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        print(f"[metrics] scoring {candidate_index}/{len(candidates)}: {candidate.name}", flush=True)

        for segment_id, (hypothesis, reference) in enumerate(
            zip(candidate.lines, reference_lines),
            start=1,
        ):
            metric_values = {
                metric_name: metric_fn(hypothesis, reference)
                for metric_name, metric_fn in METRICS.items()
            }

            rows.append(
                {
                    "candidate": candidate.name,
                    "candidate_type": candidate.candidate_type,
                    "segment_id": segment_id,
                    **metric_values,
                }
            )

    return pd.DataFrame(rows)


def aggregate_scores(segment_scores: pd.DataFrame, reference_label: str) -> pd.DataFrame:
    metric_cols = list(METRICS.keys())

    aggregated = (
        segment_scores
        .groupby(["candidate", "candidate_type"], as_index=False)[metric_cols]
        .mean()
    )

    aggregated.insert(0, "reference", reference_label)

    for metric in metric_cols:
        aggregated[f"rank_{metric}"] = (
            aggregated[metric]
            .rank(method="min", ascending=False)
            .astype(int)
        )

    return aggregated.sort_values([f"rank_{MAIN_METRICS[0]}", "candidate"])


def compute_domain_scores(segment_scores: pd.DataFrame, metadata_rows: list[dict]) -> pd.DataFrame:
    domain_df = pd.DataFrame(
        {
            "segment_id": range(1, len(metadata_rows) + 1),
            "domain": [row.get("domain", "UNKNOWN") for row in metadata_rows],
        }
    )

    enriched = segment_scores.merge(domain_df, on="segment_id", how="left")

    return (
        enriched
        .groupby(["domain", "candidate", "candidate_type"], as_index=False)[list(METRICS.keys())]
        .mean()
        .sort_values(["domain", "candidate_type", "candidate"])
    )


def mean_metric(candidate_lines: list[str], reference_lines: list[str], metric_name: str) -> float:
    metric_fn = METRICS[metric_name]

    return sum(
        metric_fn(hypothesis, reference)
        for hypothesis, reference in zip(candidate_lines, reference_lines)
    ) / len(reference_lines)


def build_pseudo_reference_scores(
    candidates: list[Candidate],
    reference_a_lines: list[str],
    reference_b_lines: list[str],
    reference_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_lookup = {candidate.name: candidate for candidate in candidates}
    rows = []
    pseudo_reference_overview = []

    pseudo_candidates = [
        Candidate(name=MAIN_REFERENCE, candidate_type="human_reference", lines=reference_a_lines),
        Candidate(name=ALT_REFERENCE, candidate_type="human_reference", lines=reference_b_lines),
        *[candidate for candidate in candidates if candidate.candidate_type == "system"],
    ]

    for metric in MAIN_METRICS:
        system_scores = reference_scores[reference_scores["candidate_type"] == "system"].copy()
        best_row = system_scores.sort_values([metric, "candidate"], ascending=[False, True]).iloc[0]

        pseudo_ref_name = str(best_row["candidate"])
        pseudo_ref_lines = candidate_lookup[pseudo_ref_name].lines

        print(f"[pseudo] metric={metric}, pseudo_reference={pseudo_ref_name}", flush=True)

        pseudo_reference_overview.append(
            {
                "metric": metric,
                "pseudo_reference": pseudo_ref_name,
                "score_against_human_reference": float(best_row[metric]),
            }
        )

        for candidate in pseudo_candidates:
            rows.append(
                {
                    "pseudo_reference_metric": metric,
                    "pseudo_reference": pseudo_ref_name,
                    "candidate": candidate.name,
                    "candidate_type": candidate.candidate_type,
                    "is_pseudo_reference_self": candidate.name == pseudo_ref_name,
                    "score": mean_metric(candidate.lines, pseudo_ref_lines, metric),
                }
            )

    pseudo_df = pd.DataFrame(rows)

    pseudo_df["rank_score"] = (
        pseudo_df.groupby("pseudo_reference_metric")["score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return pseudo_df, pd.DataFrame(pseudo_reference_overview)


def compare_rankings(reference_scores: pd.DataFrame, pseudo_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric in MAIN_METRICS:
        ref_ranks = reference_scores[
            reference_scores["candidate_type"] == "system"
        ][["candidate", f"rank_{metric}", metric]].rename(
            columns={
                f"rank_{metric}": "rank_human_reference",
                metric: "score_human_reference",
            }
        )

        pseudo_subset = pseudo_scores[
            (pseudo_scores["pseudo_reference_metric"] == metric)
            & (pseudo_scores["candidate_type"] == "system")
        ][["candidate", "rank_score", "score", "pseudo_reference"]].rename(
            columns={
                "rank_score": "rank_pseudo_reference",
                "score": "score_pseudo_reference",
            }
        )

        merged = ref_ranks.merge(pseudo_subset, on="candidate", how="inner")
        merged["metric"] = metric
        merged["rank_delta"] = merged["rank_pseudo_reference"] - merged["rank_human_reference"]
        rows.append(merged)

    return pd.concat(rows, ignore_index=True).sort_values(["metric", "rank_human_reference"])


def compute_rank_correlations(
    ranking_comparison: pd.DataFrame,
    reference_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for metric in MAIN_METRICS:
        subset = ranking_comparison[ranking_comparison["metric"] == metric]

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
                "comparison": f"{metric}: human reference vs pseudo reference",
                "spearman_rho": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_pvalue": float(kendall.pvalue),
            }
        )

    system_scores = reference_scores[reference_scores["candidate_type"] == "system"].copy()

    spearman = spearmanr(
        system_scores[f"rank_{MAIN_METRICS[0]}"],
        system_scores[f"rank_{MAIN_METRICS[1]}"],
    )

    kendall = kendalltau(
        system_scores[f"rank_{MAIN_METRICS[0]}"],
        system_scores[f"rank_{MAIN_METRICS[1]}"],
    )

    rows.append(
        {
            "comparison": f"{MAIN_METRICS[0]} vs {MAIN_METRICS[1]} under human reference",
            "spearman_rho": float(spearman.statistic),
            "spearman_pvalue": float(spearman.pvalue),
            "kendall_tau": float(kendall.statistic),
            "kendall_pvalue": float(kendall.pvalue),
        }
    )

    return pd.DataFrame(rows)


def build_disagreement_examples(
    segment_scores: pd.DataFrame,
    candidates: list[Candidate],
    source_lines: list[str],
    reference_lines: list[str],
    max_examples: int = 30,
) -> pd.DataFrame:
    system_outputs = {
        candidate.name: candidate.lines
        for candidate in candidates
        if candidate.candidate_type == "system"
    }

    systems_only = segment_scores[segment_scores["candidate_type"] == "system"]
    rows = []

    for segment_id, group in systems_only.groupby("segment_id"):
        best_word = group.sort_values(
            [MAIN_METRICS[0], "candidate"],
            ascending=[False, True],
        ).iloc[0]

        best_char = group.sort_values(
            [MAIN_METRICS[1], "candidate"],
            ascending=[False, True],
        ).iloc[0]

        if best_word["candidate"] != best_char["candidate"]:
            word_system = str(best_word["candidate"])
            char_system = str(best_char["candidate"])
            idx = int(segment_id) - 1

            rows.append(
                {
                    "segment_id": int(segment_id),
                    "source": source_lines[idx],
                    "reference_A": reference_lines[idx],
                    "best_word_system": word_system,
                    "best_word_score": float(best_word[MAIN_METRICS[0]]),
                    "best_word_output": system_outputs[word_system][idx],
                    "best_char_system": char_system,
                    "best_char_score": float(best_char[MAIN_METRICS[1]]),
                    "best_char_output": system_outputs[char_system][idx],
                    "absolute_score_gap": abs(
                        float(best_word[MAIN_METRICS[0]]) - float(best_char[MAIN_METRICS[1]])
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values("absolute_score_gap", ascending=False).head(max_examples)


def plot_top_metric_scores(reference_scores: pd.DataFrame) -> None:
    systems = reference_scores[reference_scores["candidate_type"] == "system"].copy()

    for metric in MAIN_METRICS:
        top = systems.sort_values(metric, ascending=False).head(12)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top["candidate"], top[metric])
        ax.invert_yaxis()
        ax.set_xlabel("Average segment score")
        ax.set_ylabel("System")
        ax.set_title(f"Top systems by {metric} against {MAIN_REFERENCE}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"top_systems_{metric}.png", dpi=200)
        plt.close(fig)


def plot_metric_rank_scatter(reference_scores: pd.DataFrame) -> None:
    systems = reference_scores[reference_scores["candidate_type"] == "system"].copy()
    x_metric, y_metric = MAIN_METRICS

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(systems[f"rank_{x_metric}"], systems[f"rank_{y_metric}"])

    for _, row in systems.iterrows():
        ax.annotate(
            str(row["candidate"]),
            (row[f"rank_{x_metric}"], row[f"rank_{y_metric}"]),
            fontsize=7,
        )

    ax.set_xlabel(f"Rank by {x_metric}")
    ax.set_ylabel(f"Rank by {y_metric}")
    ax.set_title("System ranking comparison under human reference")
    ax.invert_xaxis()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ranking_scatter_main_metrics.png", dpi=200)
    plt.close(fig)


def plot_rank_delta(ranking_comparison: pd.DataFrame) -> None:
    for metric in MAIN_METRICS:
        subset = ranking_comparison[ranking_comparison["metric"] == metric].copy()
        subset = subset.sort_values("rank_delta", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(subset["candidate"], subset["rank_delta"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Rank change: pseudo reference rank - human reference rank")
        ax.set_ylabel("System")
        ax.set_title(f"Ranking changes for {metric}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"rank_delta_{metric}.png", dpi=200)
        plt.close(fig)


def write_method_notes() -> None:
    notes = """# Metric method notes

This project uses reference-based n-gram similarity metrics for machine translation evaluation.
Scores are calculated segment-wise and then averaged across all 998 segments, matching the project requirement to report the overall metric value as an average over documents/lines.

## Main metrics

- `word_1_4_f1`: average F1 overlap of word n-grams from n=1 to n=4. Tokenization lowercases text and separates punctuation from word tokens.
- `char_1_6_chrf_like`: chrF-inspired character n-gram F-score from n=1 to n=6 with beta=2, giving recall more weight.

## Reference handling

- `reference_A` is used as the main human reference.
- `reference_B` is evaluated as an additional human reference candidate to expose the reference-dependence of n-gram metrics.
- For each main metric, the best machine translation according to `reference_A` is used as a pseudo-reference. All systems plus both human references are then re-evaluated against this pseudo-reference.
"""

    (TABLE_DIR / "metric_method_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    source_lines = read_lines(RAW_DIR / "source.txt")
    reference_a_lines = read_lines(RAW_DIR / "reference_A.txt")
    reference_b_lines = read_lines(RAW_DIR / "reference_B.txt")
    metadata_rows = read_jsonl(RAW_DIR / "metadata.jsonl")
    candidates = load_candidates(reference_b_lines)

    validate_equal_lengths(
        {
            "source": source_lines,
            MAIN_REFERENCE: reference_a_lines,
            ALT_REFERENCE: reference_b_lines,
            "metadata": [json.dumps(row) for row in metadata_rows],
            **{candidate.name: candidate.lines for candidate in candidates},
        }
    )

    reference_segment_scores = build_segment_scores(candidates, reference_a_lines)
    reference_scores = aggregate_scores(reference_segment_scores, MAIN_REFERENCE)
    domain_scores = compute_domain_scores(reference_segment_scores, metadata_rows)

    pseudo_scores, pseudo_reference_overview = build_pseudo_reference_scores(
        candidates=candidates,
        reference_a_lines=reference_a_lines,
        reference_b_lines=reference_b_lines,
        reference_scores=reference_scores,
    )

    ranking_comparison = compare_rankings(reference_scores, pseudo_scores)
    rank_correlations = compute_rank_correlations(ranking_comparison, reference_scores)

    disagreement_examples = build_disagreement_examples(
        reference_segment_scores,
        candidates,
        source_lines,
        reference_a_lines,
    )

    reference_segment_scores.to_csv(TABLE_DIR / "segment_scores_reference_A.csv", index=False)
    reference_scores.to_csv(TABLE_DIR / "metric_scores_reference_A.csv", index=False)
    domain_scores.to_csv(TABLE_DIR / "domain_metric_scores_reference_A.csv", index=False)
    pseudo_scores.to_csv(TABLE_DIR / "metric_scores_pseudoref.csv", index=False)
    pseudo_reference_overview.to_csv(TABLE_DIR / "pseudo_reference_overview.csv", index=False)
    ranking_comparison.to_csv(TABLE_DIR / "ranking_comparison.csv", index=False)
    rank_correlations.to_csv(TABLE_DIR / "rank_correlations.csv", index=False)
    disagreement_examples.to_csv(TABLE_DIR / "metric_disagreement_examples.csv", index=False)
    write_method_notes()

    plot_top_metric_scores(reference_scores)
    plot_metric_rank_scatter(reference_scores)
    plot_rank_delta(ranking_comparison)

    print("\nMetric scores against reference_A")
    print(
        reference_scores[
            ["candidate", "candidate_type", *MAIN_METRICS, *(f"rank_{m}" for m in MAIN_METRICS)]
        ].sort_values(f"rank_{MAIN_METRICS[0]}").to_string(index=False)
    )

    print("\nPseudo-reference overview")
    print(pseudo_reference_overview.to_string(index=False))

    print("\nRank correlations")
    print(rank_correlations.to_string(index=False))

    print("\nWrote metric outputs to:")
    print(TABLE_DIR)
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()