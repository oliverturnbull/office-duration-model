"""Generate an HTML model-monitoring dashboard using Plotly."""

import os

import numpy as np
import pandas as pd


def generate_dashboard(
    metrics_path: str = "monitoring/metrics.jsonl",
    output_path: str = "monitoring/dashboard.html",
) -> str:
    """Generate a self-contained HTML monitoring dashboard.

    Reads historical metrics from *metrics_path* and writes a Plotly-based
    HTML page to *output_path*.

    Parameters
    ----------
    metrics_path : Path to the JSONL file written by ``save_metrics``.
    output_path : Destination path for the HTML dashboard.

    Returns
    -------
    Absolute path to the generated file.

    Raises
    ------
    ImportError  if plotly is not installed.
    ValueError   if no metrics rows are found.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for dashboard generation. "
            "Install it with:  pip install plotly"
        ) from exc

    from office_duration.monitoring import load_metrics

    df = load_metrics(metrics_path)
    if df.empty:
        raise ValueError(
            f"No metrics found at '{metrics_path}'. "
            "Run the model with --monitor first."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    # Use the human-readable label on the x-axis when present
    x = df["label"] if "label" in df.columns else df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    # ------------------------------------------------------------------ #
    # Build subplot grid                                                   #
    # ------------------------------------------------------------------ #
    subplot_titles = [
        "Concordance Index (C-index) over Time",
        "AIC over Time",
        "MAE & RMSE over Time (complete sessions)",
        "Mean Duration: Observed vs Predicted",
        "Censoring Rate over Time",
        "Bias (mean prediction error) over Time",
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )

    BLUE = "#1976D2"
    RED = "#E53935"
    GREEN = "#43A047"
    ORANGE = "#FB8C00"
    PURPLE = "#8E24AA"
    PINK = "#D81B60"

    def _line(x_vals, y_vals, name, color, dash="solid", row=1, col=1):
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=7),
            ),
            row=row,
            col=col,
        )

    # Row 1 col 1 — Concordance Index
    if "concordance_index" in df.columns:
        _line(x, df["concordance_index"], "C-index", BLUE, row=1, col=1)
        fig.add_hline(
            y=0.5,
            line_dash="dot",
            line_color="gray",
            row=1,
            col=1,
            annotation_text="random baseline",
            annotation_position="top right",
            annotation_font_size=10,
        )

    # Row 1 col 2 — AIC
    if "aic" in df.columns:
        _line(x, df["aic"], "AIC", RED, row=1, col=2)

    # Row 2 col 1 — MAE and RMSE
    if "mae" in df.columns:
        _line(x, df["mae"], "MAE (h)", GREEN, row=2, col=1)
    if "rmse" in df.columns:
        _line(x, df["rmse"], "RMSE (h)", ORANGE, row=2, col=1)

    # Row 2 col 2 — Mean observed vs predicted
    if "mean_observed" in df.columns:
        _line(x, df["mean_observed"], "Observed mean", BLUE, row=2, col=2)
    if "mean_predicted" in df.columns:
        _line(x, df["mean_predicted"], "Predicted mean", PINK, dash="dash", row=2, col=2)

    # Row 3 col 1 — Censoring rate
    if "censoring_rate" in df.columns:
        _line(x, df["censoring_rate"] * 100, "Censoring %", PURPLE, row=3, col=1)

    # Row 3 col 2 — Bias
    if "bias" in df.columns:
        _line(x, df["bias"], "Bias (h)", RED, row=3, col=2)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=2)

    # ------------------------------------------------------------------ #
    # Axis labels                                                          #
    # ------------------------------------------------------------------ #
    fig.update_yaxes(title_text="C-index", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="AIC", row=1, col=2)
    fig.update_yaxes(title_text="Hours", row=2, col=1)
    fig.update_yaxes(title_text="Hours", row=2, col=2)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="Hours", row=3, col=2)

    # ------------------------------------------------------------------ #
    # Summary subtitle from the most-recent run                           #
    # ------------------------------------------------------------------ #
    latest = df.iloc[-1]
    summary_parts = []
    for key, fmt, label in [
        ("concordance_index", ".3f", "C-index"),
        ("aic", ".1f", "AIC"),
        ("mae", ".2f", "MAE"),
        ("rmse", ".2f", "RMSE"),
        ("censoring_rate", ".0%", "censored"),
        ("n_sessions", "d", "sessions"),
    ]:
        if key in latest and pd.notna(latest[key]):
            val = latest[key]
            if fmt == ".0%":
                formatted = f"{val:.0%}"
            elif fmt == "d":
                formatted = str(int(val))
            else:
                formatted = format(val, fmt)
            summary_parts.append(f"{label}: {formatted}")

    run_label = latest.get("label", str(latest.get("run_id", "N/A")))
    subtitle = "  |  ".join(summary_parts)

    # ------------------------------------------------------------------ #
    # Layout                                                               #
    # ------------------------------------------------------------------ #
    fig.update_layout(
        title=dict(
            text=(
                "Office Duration Model — Monitoring Dashboard"
                f"<br><sup>Latest run: <b>{run_label}</b>   {subtitle}</sup>"
            ),
            font=dict(size=17),
            x=0.02,
            xanchor="left",
        ),
        height=920,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        paper_bgcolor="#F5F5F5",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(t=120, b=60, l=60, r=40),
    )

    for ann in fig.layout.annotations:
        ann.font.size = 13

    # ------------------------------------------------------------------ #
    # Write output                                                         #
    # ------------------------------------------------------------------ #
    dirpart = os.path.dirname(output_path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs="cdn")
    return os.path.abspath(output_path)
