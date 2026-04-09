"""Pair raw tap events into office sessions."""

import pandas as pd
import numpy as np


def build_sessions(taps: pd.DataFrame, max_duration_hours: float = 16.0) -> pd.DataFrame:
    """
    Convert a dataframe of tap events into sessions (one row per visit).

    Each tap-in is paired with the next tap-out for the same person.
    If no tap-out is found before the next tap-in (or end of data),
    the session is marked as censored.

    Parameters
    ----------
    taps : DataFrame with columns [person_id, timestamp, direction]
    max_duration_hours : Cap for censored session durations. Used as the
        observed duration for right-censoring.

    Returns
    -------
    DataFrame with columns:
        person_id, tap_in, tap_out, censored, duration_hours
    """
    taps = taps.copy()
    taps["timestamp"] = pd.to_datetime(taps["timestamp"])
    taps = taps.sort_values(["person_id", "timestamp"]).reset_index(drop=True)

    sessions = []

    for person_id, group in taps.groupby("person_id"):
        pending_in = None

        for _, row in group.iterrows():
            if row["direction"] == "in":
                # If there's already a pending tap-in, close it as censored
                if pending_in is not None:
                    duration = (row["timestamp"] - pending_in).total_seconds() / 3600
                    sessions.append({
                        "person_id": person_id,
                        "tap_in": pending_in,
                        "tap_out": pd.NaT,
                        "censored": True,
                        "duration_hours": min(duration, max_duration_hours),
                    })
                pending_in = row["timestamp"]

            elif row["direction"] == "out" and pending_in is not None:
                duration = (row["timestamp"] - pending_in).total_seconds() / 3600
                sessions.append({
                    "person_id": person_id,
                    "tap_in": pending_in,
                    "tap_out": row["timestamp"],
                    "censored": False,
                    "duration_hours": duration,
                })
                pending_in = None

        # End of data with a pending tap-in -> censored
        if pending_in is not None:
            sessions.append({
                "person_id": person_id,
                "tap_in": pending_in,
                "tap_out": pd.NaT,
                "censored": True,
                "duration_hours": max_duration_hours,
            })

    df = pd.DataFrame(sessions)
    # Drop implausible sessions
    df = df[df["duration_hours"] > 0].reset_index(drop=True)
    return df
