"""Feature engineering for the survival model."""

import pandas as pd
import numpy as np


def add_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """Add covariates used by the survival model.

    Features:
        arrival_hour: Hour of day (0-23) of tap-in
        day_of_week: 0=Monday .. 6=Sunday
        is_weekend: Boolean
        person_freq: Total session count for this person (proxy for regularity)
    """
    df = sessions.copy()
    df["arrival_hour"] = df["tap_in"].dt.hour
    df["day_of_week"] = df["tap_in"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    freq = df.groupby("person_id").size().rename("person_freq")
    df = df.merge(freq, on="person_id", how="left")

    return df
