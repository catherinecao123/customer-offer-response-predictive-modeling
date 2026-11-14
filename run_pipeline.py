#!/usr/bin/env python3
"""End-to-end pipeline for the Customer Offer Response project.

Loads the raw data (from ``data.zip`` by default), cleans it, builds the
master table, computes descriptive findings, trains the logistic
regression and decision tree classifiers, and writes a JSON report plus
figures to the output directory.

Usage:
    python run_pipeline.py
    python run_pipeline.py --sample-fraction 0.05 --output-dir outputs/smoke
    python run_pipeline.py --data data/ --skip-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from offer_response import analysis, cleaning, config, data, features, modeling, plotting


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(config.DEFAULT_DATA_SOURCE),
        help="data.zip archive or extracted data directory (default: data.zip)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.DEFAULT_OUTPUT_DIR),
        help="directory for report.json and figures (default: outputs/)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="subsample this fraction of transcript rows (smoke runs only; "
        "metrics from a subsample are not comparable to a full run)",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="do not generate figures"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Loading raw data from {args.data}")
    portfolio_raw, profile_raw, transcript_raw = data.load_raw(args.data)
    _log(
        f"Loaded: {len(portfolio_raw)} offers, {len(profile_raw):,} customers, "
        f"{len(transcript_raw):,} events"
    )

    if args.sample_fraction is not None:
        transcript_raw = data.sample_transcript(transcript_raw, args.sample_fraction)
        _log(f"Subsampled transcript to {len(transcript_raw):,} events "
             f"({args.sample_fraction:.0%} — smoke run)")

    _log("Cleaning tables")
    portfolio = cleaning.clean_portfolio(portfolio_raw)
    profile = cleaning.clean_profile(profile_raw)
    transcript = cleaning.parse_transcript(transcript_raw)
    _log(f"Customers with unknown demographics (age 118): "
         f"{cleaning.count_missing_demographics(profile):,}")

    _log("Building master table")
    master = features.build_master_table(transcript, portfolio, profile)
    _log(f"Master table: {master.shape[0]:,} rows x {master.shape[1]} columns")

    _log("Computing descriptive findings")
    findings = analysis.summarize(master, portfolio, profile, transcript)

    _log("Building modeling frame")
    X, y, _frame = features.build_modeling_frame(master)
    _log(f"Modeling frame: {X.shape[0]:,} rows x {X.shape[1]} features, "
         f"positive rate {y.mean():.3f}")

    _log("Training and evaluating models")
    _models, metrics, split_info = modeling.train_and_evaluate(X, y)

    report = {
        "findings": findings,
        "modeling": {"split": split_info, "metrics": metrics},
        "config": {
            "random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            "features": config.FEATURE_COLUMNS,
            "sample_fraction": args.sample_fraction,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _log(f"Wrote {report_path}")

    if not args.skip_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plotting.plot_event_counts(transcript, figures_dir / "event_counts.png")
        plotting.plot_completion_by_offer_type(master, figures_dir / "completion_by_offer_type.png")
        plotting.plot_income_by_gender(profile, figures_dir / "income_by_gender.png")
        _log(f"Wrote figures to {figures_dir}")

    _log("Model metrics (test set):")
    for name, m in metrics.items():
        _log(
            f"  {name:20s} jaccard={m['jaccard']:.3f}  "
            f"f1_weighted={m['f1_weighted']:.3f}  log_loss={m['log_loss']:.3f}  "
            f"accuracy={m['accuracy']:.3f}  roc_auc={m['roc_auc']:.3f}"
        )
    _log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
