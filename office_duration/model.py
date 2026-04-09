"""Survival model fitting and prediction."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter


def fit_cox(sessions: pd.DataFrame) -> CoxPHFitter:
    """Fit a Cox proportional hazards model.

    Parameters
    ----------
    sessions : DataFrame with columns duration_hours, censored,
               plus covariate columns.

    Returns
    -------
    Fitted CoxPHFitter
    """
    covariates = ["arrival_hour", "day_of_week", "is_weekend", "person_freq"]
    df = sessions[covariates + ["duration_hours", "censored"]].copy()
    # lifelines expects event_observed (1 = event happened = not censored)
    df["event_observed"] = (~df["censored"]).astype(int)
    df = df.drop(columns=["censored"])

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="duration_hours", event_col="event_observed")
    return cph


def fit_weibull(sessions: pd.DataFrame) -> WeibullAFTFitter:
    """Fit a Weibull accelerated failure time model.

    This is useful for directly predicting expected durations,
    since AFT models give a natural E[T|X].
    """
    covariates = ["arrival_hour", "day_of_week", "is_weekend", "person_freq"]
    df = sessions[covariates + ["duration_hours", "censored"]].copy()
    df["event_observed"] = (~df["censored"]).astype(int)
    df = df.drop(columns=["censored"])

    aft = WeibullAFTFitter(penalizer=0.01)
    aft.fit(df, duration_col="duration_hours", event_col="event_observed")
    return aft


def predict_durations(model, sessions: pd.DataFrame) -> pd.Series:
    """Predict expected duration for each session.

    For a WeibullAFTFitter this returns the conditional mean.
    For a CoxPHFitter this returns the median survival time.
    """
    covariates = ["arrival_hour", "day_of_week", "is_weekend", "person_freq"]
    X = sessions[covariates]

    if isinstance(model, WeibullAFTFitter):
        return model.predict_expectation(X)
    else:
        return model.predict_median(X)
