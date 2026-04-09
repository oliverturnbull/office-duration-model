"""Generate synthetic tap data for testing."""

import argparse
import os
import pandas as pd
import numpy as np


def generate_taps(
    n_people: int = 50,
    n_days: int = 30,
    tailgate_rate: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic access card tap events.

    Parameters
    ----------
    n_people : Number of distinct people
    n_days : Number of working days to simulate
    tailgate_rate : Probability that a person tailgates out (no tap-out)
    seed : Random seed

    Returns
    -------
    DataFrame with columns [person_id, timestamp, direction]
    """
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2025-01-06")  # A Monday
    records = []

    for person in range(n_people):
        person_id = f"P{person:03d}"
        # Each person has a typical arrival hour and duration
        mean_arrival = rng.uniform(7.5, 10.5)
        mean_duration = rng.uniform(6.0, 10.0)

        for day in range(n_days):
            date = base_date + pd.Timedelta(days=day)
            if date.dayofweek >= 5:
                # ~20% chance of weekend work
                if rng.random() > 0.2:
                    continue

            # Skip some days randomly (sick, WFH)
            if rng.random() < 0.15:
                continue

            arrival_hour = rng.normal(mean_arrival, 0.5)
            arrival_hour = np.clip(arrival_hour, 6.0, 12.0)
            duration = rng.normal(mean_duration, 1.5)
            duration = np.clip(duration, 1.0, 14.0)

            tap_in_time = date + pd.Timedelta(hours=arrival_hour)
            tap_out_time = tap_in_time + pd.Timedelta(hours=duration)

            records.append({
                "person_id": person_id,
                "timestamp": tap_in_time,
                "direction": "in",
            })

            # Tailgate out with some probability
            if rng.random() > tailgate_rate:
                records.append({
                    "person_id": person_id,
                    "timestamp": tap_out_time,
                    "direction": "out",
                })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


def generate_batches(
    n_batches: int = 8,
    batch_days: int = 14,
    n_people: int = 50,
    tailgate_rate: float = 0.2,
    drift: bool = True,
    seed: int = 42,
) -> list:
    """Generate multiple sequential time-period batches for monitoring simulation.

    Each batch represents *batch_days* calendar days.  When *drift* is True
    the population's mean arrival time shifts later and mean stay duration
    shortens by ~5 % per batch, simulating gradual behavioural change.

    Parameters
    ----------
    n_batches : Number of batches to generate
    batch_days : Calendar days covered by each batch
    n_people : Number of distinct people
    tailgate_rate : Probability of missing tap-out per session
    drift : If True, gradually shift arrival and duration distributions
    seed : Base random seed

    Returns
    -------
    List of (label, DataFrame) tuples, one per batch.
    """
    rng = np.random.default_rng(seed)
    batches = []
    base_date = pd.Timestamp("2025-01-06")  # Monday

    # Per-person baseline characteristics (stable across batches)
    mean_arrivals = rng.uniform(7.5, 10.5, size=n_people)
    mean_durations = rng.uniform(6.0, 10.0, size=n_people)

    for batch_idx in range(n_batches):
        batch_seed = int(rng.integers(0, 2**31))
        batch_rng = np.random.default_rng(batch_seed)

        # Drift: +0.1h later arrival, -0.15h shorter duration per batch
        arrival_shift = batch_idx * 0.10 if drift else 0.0
        duration_shift = batch_idx * -0.15 if drift else 0.0

        batch_start = base_date + pd.Timedelta(days=batch_idx * batch_days)
        records = []

        for person in range(n_people):
            person_id = f"P{person:03d}"
            eff_arrival = mean_arrivals[person] + arrival_shift
            eff_duration = mean_durations[person] + duration_shift

            for day in range(batch_days):
                date = batch_start + pd.Timedelta(days=day)
                if date.dayofweek >= 5:
                    if batch_rng.random() > 0.2:
                        continue

                if batch_rng.random() < 0.15:
                    continue

                arrival_hour = batch_rng.normal(eff_arrival, 0.5)
                arrival_hour = np.clip(arrival_hour, 6.0, 12.0)
                duration = batch_rng.normal(eff_duration, 1.5)
                duration = np.clip(duration, 1.0, 14.0)

                tap_in_time = date + pd.Timedelta(hours=arrival_hour)
                tap_out_time = tap_in_time + pd.Timedelta(hours=duration)

                records.append({
                    "person_id": person_id,
                    "timestamp": tap_in_time,
                    "direction": "in",
                })

                if batch_rng.random() > tailgate_rate:
                    records.append({
                        "person_id": person_id,
                        "timestamp": tap_out_time,
                        "direction": "out",
                    })

        df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        label = f"Batch {batch_idx + 1} ({batch_start.strftime('%d %b %Y')})"
        batches.append((label, df))

    return batches


def run_monitoring_simulation(
    n_batches: int = 8,
    batch_days: int = 14,
    n_people: int = 50,
    tailgate_rate: float = 0.2,
    drift: bool = True,
    model_type: str = "weibull",
    metrics_path: str = "monitoring/metrics.jsonl",
    dashboard_path: str = "monitoring/dashboard.html",
    seed: int = 42,
    max_duration: float = 16.0,
    cumulative: bool = True,
) -> str:
    """Simulate rolling model retraining and generate a monitoring dashboard.

    Generates *n_batches* of synthetic tap data.  For each batch the full
    pipeline (build sessions → features → fit model → compute metrics) is
    run and metrics are appended to *metrics_path*.  Finally an HTML
    dashboard is written to *dashboard_path*.

    Parameters
    ----------
    n_batches : Number of time periods to simulate
    batch_days : Calendar days per batch
    n_people : Simulated headcount
    tailgate_rate : Fraction of sessions with missing tap-out
    drift : Simulate gradual distribution shift across batches
    model_type : "weibull" or "cox"
    metrics_path : JSONL file to (over)write metrics history
    dashboard_path : Output path for the HTML dashboard
    seed : Base random seed
    max_duration : Cap for censored session durations (hours)
    cumulative : If True fit on all data so far (growing dataset); if False
        fit on each batch independently (rolling window).

    Returns
    -------
    Path to the generated HTML dashboard.
    """
    from office_duration.sessions import build_sessions
    from office_duration.features import add_features
    from office_duration.model import fit_weibull, fit_cox, predict_durations
    from office_duration.monitoring import compute_metrics, save_metrics
    from office_duration.dashboard import generate_dashboard

    if model_type == "weibull":
        fit_fn = fit_weibull
    else:
        fit_fn = fit_cox

    # Clear any existing metrics file so each simulation starts fresh
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    print(f"Simulating {n_batches} batches × {batch_days} days "
          f"({'cumulative' if cumulative else 'rolling'} training, "
          f"drift={'on' if drift else 'off'})…")

    batches = generate_batches(
        n_batches=n_batches,
        batch_days=batch_days,
        n_people=n_people,
        tailgate_rate=tailgate_rate,
        drift=drift,
        seed=seed,
    )

    accumulated_taps = []

    for batch_idx, (label, taps_df) in enumerate(batches):
        if cumulative:
            accumulated_taps.append(taps_df)
            fitting_taps = pd.concat(accumulated_taps, ignore_index=True)
        else:
            fitting_taps = taps_df

        sessions = build_sessions(fitting_taps, max_duration_hours=max_duration)
        sessions = add_features(sessions)

        if len(sessions) < 10:
            print(f"  [{label}] skipped (too few sessions: {len(sessions)})")
            continue

        try:
            model = fit_fn(sessions)
        except Exception as exc:
            print(f"  [{label}] model fitting failed: {exc}")
            continue

        metrics = compute_metrics(model, sessions, label=label)
        save_metrics(metrics, path=metrics_path)

        line_parts = [f"C-index={metrics['concordance_index']:.3f}",
                      f"AIC={metrics['aic']:.1f}"]
        if "mae" in metrics:
            line_parts.append(f"MAE={metrics['mae']:.2f}h")
        print(f"  [{label}]  {',  '.join(line_parts)}")

    print(f"\nMetrics saved to {metrics_path}")
    print("Generating dashboard…")
    out = generate_dashboard(metrics_path=metrics_path, output_path=dashboard_path)
    print(f"Dashboard written to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tap data")
    parser.add_argument("--people", type=int, default=50)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--tailgate-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/taps.csv")

    # Monitoring simulation subcommand flags
    monitor_group = parser.add_argument_group(
        "monitoring simulation",
        "Generate multiple batches and produce a monitoring dashboard.",
    )
    monitor_group.add_argument(
        "--simulate-monitoring",
        action="store_true",
        help="Run a multi-batch monitoring simulation instead of single-file generation",
    )
    monitor_group.add_argument(
        "--batches",
        type=int,
        default=8,
        help="Number of time-period batches to simulate (default: 8)",
    )
    monitor_group.add_argument(
        "--batch-days",
        type=int,
        default=14,
        help="Calendar days per batch (default: 14)",
    )
    monitor_group.add_argument(
        "--no-drift",
        action="store_true",
        help="Disable distribution drift across batches",
    )
    monitor_group.add_argument(
        "--rolling",
        action="store_true",
        help="Use a rolling window (each batch only) instead of cumulative training data",
    )
    monitor_group.add_argument(
        "--model",
        choices=["weibull", "cox"],
        default="weibull",
        help="Survival model type for monitoring simulation (default: weibull)",
    )
    monitor_group.add_argument(
        "--metrics-path",
        default="monitoring/metrics.jsonl",
        help="JSONL output path for metrics (default: monitoring/metrics.jsonl)",
    )
    monitor_group.add_argument(
        "--dashboard-path",
        default="monitoring/dashboard.html",
        help="HTML output path for dashboard (default: monitoring/dashboard.html)",
    )

    args = parser.parse_args()

    if args.simulate_monitoring:
        run_monitoring_simulation(
            n_batches=args.batches,
            batch_days=args.batch_days,
            n_people=args.people,
            tailgate_rate=args.tailgate_rate,
            drift=not args.no_drift,
            model_type=args.model,
            metrics_path=args.metrics_path,
            dashboard_path=args.dashboard_path,
            seed=args.seed,
            cumulative=not args.rolling,
        )
    else:
        df = generate_taps(args.people, args.days, args.tailgate_rate, args.seed)
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Generated {len(df)} tap events for {args.people} people "
              f"over {args.days} days -> {args.output}")


if __name__ == "__main__":
    main()
