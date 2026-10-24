"""Print the offline structural value-model validation case study as Markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_signals.analysis.value_model_case_study import (  # noqa: E402
    DEFAULT_FEATURE_MATRIX_PATH,
    render_value_model_case_study,
    run_value_model_case_study,
)
from market_signals.models.value_regression import MIN_TRAINING_SAMPLES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=DEFAULT_FEATURE_MATRIX_PATH,
        help="Saved feature-matrix CSV; defaults to the committed master matrix.",
    )
    parser.add_argument(
        "--min-training-samples",
        type=int,
        default=MIN_TRAINING_SAMPLES,
        help=(
            f"Minimum eligible rows required to train; project default is {MIN_TRAINING_SAMPLES}, "
            "with a floor of 9 for three-way evaluation."
        ),
    )
    args = parser.parse_args()
    if args.min_training_samples < 1:
        parser.error("--min-training-samples must be at least 1")
    report = run_value_model_case_study(
        args.feature_matrix,
        min_training_samples=args.min_training_samples,
    )
    print(render_value_model_case_study(report), end="")


if __name__ == "__main__":
    main()
