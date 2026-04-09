"""Generate synthetic tap data for testing."""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tap data")
    parser.add_argument("--people", type=int, default=50)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--tailgate-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/taps.csv")
    args = parser.parse_args()

    df = generate_taps(args.people, args.days, args.tailgate_rate, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} tap events for {args.people} people over {args.days} days -> {args.output}")


if __name__ == "__main__":
    main()
