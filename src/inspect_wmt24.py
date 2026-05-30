from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


LANG_PAIR = "en-de"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wmt24" / LANG_PAIR
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def simple_word_count(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest() -> dict[str, Any]:
    manifest_path = RAW_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    source_path = RAW_DIR / "source.txt"
    ref_a_path = RAW_DIR / "reference_A.txt"
    ref_b_path = RAW_DIR / "reference_B.txt"
    metadata_path = RAW_DIR / "metadata.jsonl"
    system_dir = RAW_DIR / "system_outputs"

    required_paths = [source_path, ref_a_path, ref_b_path, metadata_path, system_dir]
    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        raise RuntimeError(f"Missing required data paths: {missing}")

    source_lines = read_lines(source_path)
    ref_a_lines = read_lines(ref_a_path)
    ref_b_lines = read_lines(ref_b_path)
    metadata_rows = read_jsonl(metadata_path)

    system_files = sorted(system_dir.glob("*.txt"))
    if not system_files:
        raise RuntimeError("No system outputs found. Run download_wmt24.py first.")

    expected_segments = len(source_lines)

    system_counts = []
    incomplete_systems = []

    for path in system_files:
        lines = read_lines(path)
        complete = len(lines) == expected_segments

        if not complete:
            incomplete_systems.append(path.stem)

        system_counts.append({
            "system": path.stem,
            "segments": len(lines),
            "target_words": simple_word_count(lines),
            "complete": complete,
        })

    system_df = pd.DataFrame(system_counts).sort_values("system")

    manifest = load_manifest()
    download_summary = manifest.get("download_summary", {})

    data_summary = pd.DataFrame([
        {
            "language_pair": LANG_PAIR,
            "source_segments": len(source_lines),
            "reference_A_segments": len(ref_a_lines),
            "reference_B_segments": len(ref_b_lines),
            "metadata_rows": len(metadata_rows),
            "source_words": simple_word_count(source_lines),
            "reference_A_words": simple_word_count(ref_a_lines),
            "reference_B_words": simple_word_count(ref_b_lines),
            "system_count_local": len(system_files),
            "system_count_expected": download_summary.get("expected_system_outputs", "unknown"),
            "complete_system_outputs": int(system_df["complete"].sum()),
            "download_passed": download_summary.get("passed", "unknown"),
        }
    ])

    metadata_info = {
        "metadata_rows": len(metadata_rows),
        "metadata_keys": sorted(metadata_rows[0].keys()) if metadata_rows else [],
    }

    domain_summary = pd.DataFrame()
    detected_domain_key = None

    for key in ["domain", "doc_domain", "genre", "source_domain"]:
        if metadata_rows and key in metadata_rows[0]:
            detected_domain_key = key
            break

    if detected_domain_key:
        domain_counts = Counter(row.get(detected_domain_key, "UNKNOWN") for row in metadata_rows)
        domain_summary = pd.DataFrame([
            {
                "domain": domain,
                "segments": count,
            }
            for domain, count in sorted(domain_counts.items())
        ])

    data_summary.to_csv(TABLE_DIR / "data_summary.csv", index=False)
    system_df.to_csv(TABLE_DIR / "system_outputs_summary.csv", index=False)

    if not domain_summary.empty:
        domain_summary.to_csv(TABLE_DIR / "domain_summary.csv", index=False)

    metadata_report_path = TABLE_DIR / "metadata_report.json"
    metadata_report_path.write_text(
        json.dumps(metadata_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nData summary")
    print(data_summary.to_string(index=False))

    print("\nSystem outputs summary")
    print(system_df.to_string(index=False))

    print("\nMetadata info")
    print(json.dumps(metadata_info, indent=2, ensure_ascii=False))

    if not domain_summary.empty:
        print("\nDomain summary")
        print(domain_summary.to_string(index=False))

    if incomplete_systems:
        raise RuntimeError(
            f"Incomplete system output files detected: {incomplete_systems}. "
            "Re-run download_wmt24.py."
        )


if __name__ == "__main__":
    main()