from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


LANG_PAIR_DEFAULT = "en-de"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "wmt-conference/wmt24-news-systems/main/txt"
)
GITHUB_API_BASE = (
    "https://api.github.com/repos/wmt-conference/wmt24-news-systems/"
    "contents/txt/system-outputs"
)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def request_url(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "nlp-mt-metrics-project/1.0",
            "Accept": "application/vnd.github+json, text/plain, */*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_with_retries(
    url: str,
    target_path: Path,
    retries: int = 5,
    sleep_seconds: float = 2.0,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            print(f"[download] attempt {attempt}/{retries}: {url}")
            content = request_url(url)
            temp_path.write_bytes(content)
            temp_path.replace(target_path)
            print(f"[saved] {target_path}")
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            last_error = error
            print(f"[warning] download failed: {error}")
            if attempt < retries:
                wait_time = sleep_seconds * attempt
                print(f"[retry] waiting {wait_time:.1f}s")
                time.sleep(wait_time)

    raise RuntimeError(f"Download failed after {retries} attempts: {url}") from last_error


def read_json_from_url(url: str) -> list[dict[str, Any]]:
    content = request_url(url)
    return json.loads(content.decode("utf-8"))


def file_is_complete(path: Path, expected_lines: int | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if expected_lines is None:
        return True
    return count_lines(path) == expected_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-pair", default=LANG_PAIR_DEFAULT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    lang_pair = args.lang_pair
    raw_dir = PROJECT_ROOT / "data" / "raw" / "wmt24" / lang_pair
    raw_dir.mkdir(parents=True, exist_ok=True)

    fixed_files = {
        "source": f"{RAW_BASE}/sources/{lang_pair}.txt",
        "reference_A": f"{RAW_BASE}/references/{lang_pair}.refA.txt",
        "reference_B": f"{RAW_BASE}/references/{lang_pair}.refB.txt",
        "metadata": f"{RAW_BASE}/metadata/{lang_pair}.jsonl",
    }

    manifest: dict[str, Any] = {
        "language_pair": lang_pair,
        "fixed_files": {},
        "system_outputs": {},
        "failures": [],
    }

    # 1) Download fixed files first.
    for name, url in fixed_files.items():
        suffix = ".jsonl" if name == "metadata" else ".txt"
        target_path = raw_dir / f"{name}{suffix}"

        if not args.force and file_is_complete(target_path, expected_lines=None):
            print(f"[skip] existing file: {target_path}")
        else:
            download_with_retries(url, target_path)

        manifest["fixed_files"][name] = {
            "url": url,
            "local_path": str(target_path.relative_to(PROJECT_ROOT)),
            "line_count": count_lines(target_path),
        }

    expected_lines = manifest["fixed_files"]["source"]["line_count"]
    print(f"[info] expected segment count: {expected_lines}")

    # 2) Discover system output files via GitHub API.
    api_url = f"{GITHUB_API_BASE}/{lang_pair}"
    system_items = read_json_from_url(api_url)
    system_files = [
        item for item in system_items
        if item.get("type") == "file" and item.get("name", "").endswith(".txt")
    ]

    if not system_files:
        raise RuntimeError(f"No system output files found for {lang_pair}.")

    print(f"[info] discovered system outputs: {len(system_files)}")

    # 3) Download all system outputs with resume behavior.
    for item in system_files:
        system_name = item["name"].removesuffix(".txt")
        target_path = raw_dir / "system_outputs" / item["name"]

        try:
            if not args.force and file_is_complete(target_path, expected_lines=expected_lines):
                print(f"[skip] complete system output: {target_path.name}")
            else:
                download_with_retries(item["download_url"], target_path)

            line_count = count_lines(target_path)

            manifest["system_outputs"][system_name] = {
                "url": item["download_url"],
                "local_path": str(target_path.relative_to(PROJECT_ROOT)),
                "line_count": line_count,
                "complete": line_count == expected_lines,
            }

            if line_count != expected_lines:
                manifest["failures"].append({
                    "system": system_name,
                    "reason": f"line_count={line_count}, expected={expected_lines}",
                })

        except Exception as error:
            manifest["failures"].append({
                "system": system_name,
                "reason": repr(error),
            })
            print(f"[error] failed system output {system_name}: {error}")

    txt_line_counts = []
    for info in manifest["fixed_files"].values():
        if info["local_path"].endswith(".txt"):
            txt_line_counts.append(info["line_count"])

    for info in manifest["system_outputs"].values():
        txt_line_counts.append(info["line_count"])

    manifest["download_summary"] = {
        "expected_system_outputs": len(system_files),
        "downloaded_system_outputs": len(manifest["system_outputs"]),
        "complete_system_outputs": sum(
            1 for item in manifest["system_outputs"].values()
            if item.get("complete")
        ),
        "unique_line_counts": sorted(set(txt_line_counts)),
        "passed": (
            len(manifest["failures"]) == 0
            and len(manifest["system_outputs"]) == len(system_files)
            and len(set(txt_line_counts)) == 1
        ),
    }

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDownload summary")
    print(json.dumps(manifest["download_summary"], indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")

    if not manifest["download_summary"]["passed"]:
        raise RuntimeError(
            "Download incomplete. Re-run the script; it will resume and retry missing files."
        )


if __name__ == "__main__":
    main()