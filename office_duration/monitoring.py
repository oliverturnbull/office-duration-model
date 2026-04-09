"""Model monitoring: compute, store, and retrieve goodness-of-fit metrics."""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def compute_metrics(
    model,
    sessions: pd.DataFrame,
    run_id: str = None,
    label: str = None,
) -> dict:
    """Compute goodness-of-fit and predictive performance metrics.

    Parameters
    ----------
    model : Fitted WeibullAFTFitter or CoxPHFitter
    sessions : DataFrame with feature columns and duration_hours/censored
    run_id : Optional run identifier (defaults to ISO UTC timestamp)
    label : Optional human-readable label for this run

    Returns
    -------
    dict of metric name -> value, plus metadata fields.
    The special key ``_predictions`` holds per-session observed/predicted
    arrays (not persisted to JSONL; used for scatter plots within a run).
    """
    from lifelines import WeibullAFTFitter
    from office_duration.model import predict_durations

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = run_id or now_iso

    metrics: dict = {
        "run_id": run_id,
        "label": label or run_id,
        "timestamp": now_iso,  # always ISO-formatted; independent of run_id
        "n_sessions": int(len(sessions)),
        "n_censored": int(sessions["censored"].sum()),
        "censoring_rate": float(sessions["censored"].mean()),
    }

    # --- Model-level fit statistics ---
    metrics["concordance_index"] = float(model.concordance_index_)
    metrics["log_likelihood"] = float(model.log_likelihood_)

    if isinstance(model, WeibullAFTFitter):
        metrics["aic"] = float(model.AIC_)
        metrics["model_type"] = "weibull"
        # Weibull shape parameter rho > 1 means increasing hazard (longer stays
        # become more likely to end), rho < 1 means decreasing hazard.
        try:
            rho_log = float(model.params_["rho_"]["Intercept"])
            metrics["weibull_rho"] = float(np.exp(rho_log))
        except (KeyError, TypeError):
            pass
    else:
        metrics["aic"] = float(model.AIC_partial_)
        metrics["model_type"] = "cox"

    # --- Predictive accuracy on complete (uncensored) sessions ---
    complete = sessions[~sessions["censored"]].copy()
    metrics["n_complete"] = int(len(complete))

    if len(complete) >= 5:
        predicted = predict_durations(model, complete).to_numpy().astype(float)
        observed = complete["duration_hours"].to_numpy().astype(float)

        residuals = predicted - observed
        abs_residuals = np.abs(residuals)

        metrics["mae"] = float(np.mean(abs_residuals))
        metrics["rmse"] = float(np.sqrt(np.mean(residuals ** 2)))
        metrics["bias"] = float(np.mean(residuals))
        metrics["mape"] = float(np.mean(abs_residuals / np.maximum(observed, 1e-6)) * 100)
        metrics["mean_observed"] = float(np.mean(observed))
        metrics["mean_predicted"] = float(np.mean(predicted))

        if np.std(predicted) > 0 and np.std(observed) > 0:
            metrics["correlation"] = float(np.corrcoef(predicted, observed)[0, 1])
        else:
            metrics["correlation"] = float("nan")

        # Not persisted to disk; used for scatter plots within a single session
        metrics["_predictions"] = {
            "observed": observed.tolist(),
            "predicted": predicted.tolist(),
        }

    return metrics


def save_metrics(metrics: dict, path: str = "monitoring/metrics.jsonl") -> None:
    """Append a metrics dict as one line to a JSONL file.

    The transient ``_predictions`` key is excluded from the file.
    """
    dirpart = os.path.dirname(path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)

    row = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_metrics(path: str = "monitoring/metrics.jsonl") -> pd.DataFrame:
    """Load all historical metrics from a JSONL file into a DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame()

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df
