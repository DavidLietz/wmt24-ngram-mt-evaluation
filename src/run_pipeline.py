from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
RUN_LOG = LOG_DIR / "run_pipeline.log"


def run_step(name: str, command: list[str]) -> int:
    header = (
        "\n"
        + "=" * 90
        + f"\nSTEP: {name}\n"
        + f"TIME: {datetime.now().isoformat(timespec='seconds')}\n"
        + f"COMMAND: {' '.join(command)}\n"
        + "=" * 90
        + "\n"
    )

    print(header)

    with RUN_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(header)

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        process.wait()

        footer = f"\nEXIT CODE: {process.returncode}\n"
        print(footer)
        log_file.write(footer)

        return process.returncode


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    RUN_LOG.write_text(
        f"Pipeline started: {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )

    steps = [
        ("download_wmt24", [sys.executable, "src/download_wmt24.py"]),
        ("inspect_wmt24", [sys.executable, "src/inspect_wmt24.py"]),
        ("evaluate_metrics", [sys.executable, "src/evaluate_metrics.py"]),
        ("build_report_assets", [sys.executable, "src/build_report_assets.py"]),
        ("build_report_supplements", [sys.executable, "src/build_report_supplements.py"]),
    ]

    for name, command in steps:
        exit_code = run_step(name, command)
        if exit_code != 0:
            print(f"\nPipeline failed in step: {name}")
            sys.exit(exit_code)

    print("\nPipeline finished successfully.")
    print("Report assets are available in: outputs/report_assets/")


if __name__ == "__main__":
    main()