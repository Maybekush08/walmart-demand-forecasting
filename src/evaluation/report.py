"""Self-contained HTML reporting for multi-model experiments."""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#4f46e5", "#db2777", "#059669", "#d97706", "#2563eb"]


def _figure_data_uri(figure: plt.Figure) -> str:
    """Encode a Matplotlib figure as an inline PNG data URI."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _comparison_chart(runs: dict[str, dict[str, Any]]) -> str | None:
    complete = [
        (name, run) for name, run in runs.items()
        if run.get("status") == "completed" and run.get("metrics")
    ]
    if not complete:
        return None

    names = [name for name, _ in complete]
    mae = [run["metrics"]["MAE"] for _, run in complete]
    rmse = [run["metrics"]["RMSE"] for _, run in complete]
    positions = np.arange(len(names))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, values, title in zip(axes, (mae, rmse), ("MAE", "RMSE")):
        bars = axis.bar(positions, values, color=COLORS[: len(names)])
        axis.set_xticks(positions, names, rotation=25, ha="right")
        axis.set_title(f"Test {title} (lower is better)")
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.tight_layout()
    return _figure_data_uri(figure)


def _training_chart(runs: dict[str, dict[str, Any]]) -> str | None:
    complete = [
        (name, run) for name, run in runs.items()
        if run.get("status") == "completed" and run.get("history")
    ]
    if not complete:
        return None

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for color, (name, run) in zip(COLORS, complete):
        history = run["history"]
        epochs = np.arange(1, len(history["val_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], label=name, color=color)
        axes[1].plot(epochs, history["learning_rate"], label=name, color=color)

    axes[0].set_title("Validation MAE by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Learning-rate schedule")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    return _figure_data_uri(figure)


def _forecast_charts(runs: dict[str, dict[str, Any]]) -> list[str]:
    complete = [
        (name, run) for name, run in runs.items()
        if run.get("status") == "completed" and run.get("sample_predictions")
    ]
    if not complete:
        return []

    sample_count = min(
        len(run["sample_predictions"]) for _, run in complete
    )
    charts = []
    for sample_index in range(sample_count):
        actual = np.asarray(complete[0][1]["sample_targets"][sample_index])
        days = np.arange(1, len(actual) + 1)
        figure, axis = plt.subplots(figsize=(11, 4.5))
        axis.plot(days, actual, color="#111827", linewidth=2.4, label="Actual")
        for color, (name, run) in zip(COLORS, complete):
            prediction = np.asarray(run["sample_predictions"][sample_index])
            axis.plot(days, prediction, color=color, linewidth=1.5, label=name)
        axis.set_title(f"Forecast comparison - held-out sample {sample_index + 1}")
        axis.set_xlabel("Forecast day")
        axis.set_ylabel("Unit sales")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
        figure.tight_layout()
        charts.append(_figure_data_uri(figure))
    return charts


def _result_rows(runs: dict[str, dict[str, Any]]) -> str:
    rows = []
    for name, run in runs.items():
        status = html.escape(str(run.get("status", "pending")))
        metrics = run.get("metrics", {})
        mae = f'{metrics["MAE"]:.4f}' if "MAE" in metrics else "-"
        rmse = f'{metrics["RMSE"]:.4f}' if "RMSE" in metrics else "-"
        best_val = run.get("best_val_mae")
        best_val_text = f"{best_val:.4f}" if best_val is not None else "-"
        seconds = run.get("duration_seconds")
        duration = f"{seconds / 60:.1f} min" if seconds is not None else "-"
        parameters = run.get("parameter_count")
        parameter_text = f"{parameters:,}" if parameters is not None else "-"
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td><td>{status}</td>"
            f"<td>{mae}</td><td>{rmse}</td><td>{best_val_text}</td>"
            f"<td>{parameter_text}</td><td>{duration}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="7">No models started.</td></tr>'


def _run_details(runs: dict[str, dict[str, Any]]) -> str:
    sections = []
    for name, run in runs.items():
        config = html.escape(json.dumps(run.get("config", {}), indent=2))
        status = html.escape(str(run.get("status", "pending")))
        body = [f"<p><strong>Status:</strong> {status}</p>"]
        if run.get("error"):
            body.append(
                '<pre class="error">'
                + html.escape(str(run["error"]))
                + "</pre>"
            )
        history = run.get("history")
        if history:
            epoch_rows = []
            for index, (train, val, rate) in enumerate(
                zip(
                    history["train_loss"],
                    history["val_loss"],
                    history["learning_rate"],
                ),
                start=1,
            ):
                epoch_rows.append(
                    f"<tr><td>{index}</td><td>{train:.6f}</td>"
                    f"<td>{val:.6f}</td><td>{rate:.8f}</td></tr>"
                )
            body.append(
                "<h4>Epoch history</h4><div class=\"table-scroll\"><table>"
                "<thead><tr><th>Epoch</th><th>Training objective</th>"
                "<th>Validation MAE</th><th>Learning rate</th></tr></thead>"
                f"<tbody>{''.join(epoch_rows)}</tbody></table></div>"
            )
        body.append(f"<h4>Configuration</h4><pre>{config}</pre>")
        sections.append(
            f"<details><summary>{html.escape(name)}</summary>{''.join(body)}</details>"
        )
    return "".join(sections)


def write_experiment_report(
    output_path: str | Path,
    runs: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Write an atomically updated, self-contained experiment HTML file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison = _comparison_chart(runs)
    training = _training_chart(runs)
    forecasts = _forecast_charts(runs)
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    chart_sections = []
    if comparison:
        chart_sections.append(
            f'<section><h2>Model comparison</h2><img src="{comparison}" '
            'alt="MAE and RMSE model comparison"></section>'
        )
    if training:
        chart_sections.append(
            f'<section><h2>Training diagnostics</h2><img src="{training}" '
            'alt="Validation and learning-rate curves"></section>'
        )
    if forecasts:
        chart_sections.append("<section><h2>Forecast samples</h2>")
        chart_sections.extend(
            f'<img src="{chart}" alt="Forecast sample comparison">'
            for chart in forecasts
        )
        chart_sections.append("</section>")

    metadata_json = html.escape(json.dumps(metadata, indent=2))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Walmart Demand Forecasting - Experiment Report</title>
<style>
body {{ margin: 0; background: #f3f4f6; color: #111827; font-family: Arial, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 22px 80px; }}
header {{ color: white; padding: 32px; border-radius: 16px; background: linear-gradient(135deg,#312e81,#4f46e5); }}
h1 {{ margin: 0 0 8px; }} h2 {{ margin-top: 34px; }} h4 {{ margin-bottom: 8px; }}
.card, section, details {{ background: white; margin-top: 20px; padding: 22px; border-radius: 12px; box-shadow: 0 2px 10px #00000012; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #eef2ff; }} img {{ display: block; max-width: 100%; margin: 15px auto; }}
summary {{ cursor: pointer; font-size: 1.1rem; font-weight: bold; }}
pre {{ padding: 14px; overflow: auto; background: #111827; color: #e5e7eb; border-radius: 8px; }}
.error {{ background: #7f1d1d; }} .table-scroll {{ overflow-x: auto; }}
.muted {{ color: #dbeafe; }}
</style>
</head>
<body><main>
<header><h1>Walmart Demand Forecasting</h1>
<p>Consolidated all-model experiment report</p>
<p class="muted">Updated {generated}. This file is self-contained; charts are embedded.</p></header>
<section><h2>Evaluation summary</h2>
<table><thead><tr><th>Model</th><th>Status</th><th>Test MAE</th><th>Test RMSE</th>
<th>Best validation MAE</th><th>Parameters</th><th>Training time</th></tr></thead>
<tbody>{_result_rows(runs)}</tbody></table></section>
{''.join(chart_sections)}
<section><h2>Run metadata</h2><pre>{metadata_json}</pre></section>
<section><h2>Per-model details</h2>{_run_details(runs)}</section>
</main></body></html>"""

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    return output
