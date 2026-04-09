"""Command-line interface for the office duration model."""

import argparse
import sys
import pandas as pd

from office_duration.sessions import build_sessions
from office_duration.features import add_features
from office_duration.model import fit_weibull, predict_durations


def main():
    parser = argparse.ArgumentParser(
        description="Estimate office visit durations from access card taps."
    )
    parser.add_argument("input", help="Path to CSV of tap events (person_id, timestamp, direction)")
    parser.add_argument("--output", "-o", default="results.csv", help="Output CSV path")
    parser.add_argument("--max-duration", type=float, default=16.0,
                        help="Max hours for censored session cap (default: 16)")
    parser.add_argument("--model", choices=["weibull", "cox"], default="weibull",
                        help="Survival model type (default: weibull)")

    # Monitoring flags
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Save goodness-of-fit and performance metrics after fitting",
    )
    parser.add_argument(
        "--metrics-path",
        default="monitoring/metrics.jsonl",
        help="Path to JSONL file for appending run metrics (default: monitoring/metrics.jsonl)",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Human-readable label for this run (stored in metrics)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate an HTML monitoring dashboard after saving metrics (implies --monitor)",
    )
    parser.add_argument(
        "--dashboard-path",
        default="monitoring/dashboard.html",
        help="Path for the HTML dashboard output (default: monitoring/dashboard.html)",
    )

    args = parser.parse_args()

    if args.dashboard:
        args.monitor = True

    # Load
    print(f"Loading taps from {args.input}")
    taps = pd.read_csv(args.input)
    print(f"  {len(taps)} tap events, {taps['person_id'].nunique()} people")

    # Build sessions
    sessions = build_sessions(taps, max_duration_hours=args.max_duration)
    n_censored = sessions["censored"].sum()
    print(f"  {len(sessions)} sessions, {n_censored} censored ({n_censored/len(sessions):.0%})")

    # Features
    sessions = add_features(sessions)

    # Fit
    if args.model == "weibull":
        from office_duration.model import fit_weibull as fit_fn
    else:
        from office_duration.model import fit_cox as fit_fn

    print(f"Fitting {args.model} model...")
    model = fit_fn(sessions)
    model.print_summary()

    # Predict
    sessions["estimated_hours"] = predict_durations(model, sessions)
    sessions["observed_hours"] = sessions.apply(
        lambda r: r["duration_hours"] if not r["censored"] else None, axis=1
    )

    # Output
    out_cols = ["person_id", "tap_in", "tap_out", "censored", "observed_hours", "estimated_hours"]
    sessions[out_cols].to_csv(args.output, index=False)
    print(f"\nResults written to {args.output}")

    # Summary stats
    complete = sessions[~sessions["censored"]]
    censored = sessions[sessions["censored"]]
    print(f"\nSummary:")
    print(f"  Complete sessions: mean {complete['duration_hours'].mean():.1f}h, "
          f"median {complete['duration_hours'].median():.1f}h")
    if len(censored) > 0:
        print(f"  Censored sessions: estimated mean {censored['estimated_hours'].mean():.1f}h, "
              f"median {censored['estimated_hours'].median():.1f}h")

    # Monitoring
    if args.monitor:
        from office_duration.monitoring import compute_metrics, save_metrics
        print(f"\nComputing monitoring metrics...")
        metrics = compute_metrics(model, sessions, label=args.run_label)
        save_metrics(metrics, path=args.metrics_path)
        print(f"  Concordance index : {metrics['concordance_index']:.4f}")
        print(f"  AIC               : {metrics['aic']:.2f}")
        if "mae" in metrics:
            print(f"  MAE (complete)    : {metrics['mae']:.3f}h")
            print(f"  RMSE (complete)   : {metrics['rmse']:.3f}h")
            print(f"  Bias              : {metrics['bias']:+.3f}h")
        print(f"  Metrics appended to {args.metrics_path}")

    if args.dashboard:
        from office_duration.dashboard import generate_dashboard
        print(f"\nGenerating monitoring dashboard...")
        out = generate_dashboard(
            metrics_path=args.metrics_path,
            output_path=args.dashboard_path,
        )
        print(f"  Dashboard written to {out}")


if __name__ == "__main__":
    main()
