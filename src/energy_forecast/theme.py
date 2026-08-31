"""Shared visual language for reports and the dashboard.

One palette, defined once. `scripts/generate_eda.py`, `scripts/benchmark.py`, and
`app/streamlit_app.py` all import from here so a chart looks the same wherever it
appears. The categorical hues are the project's validated reference palette
(CVD-checked, light surface); `MODEL_COLORS` pins each model to one hue by
identity so a legend never repaints when the set of series changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# Categorical hues — fixed order, never cycled. Validated on a light surface.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"

# Text and chrome — ink tokens for every label; chrome stays recessive.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Model identity — HistGradientBoosting is the champion and wears the primary hue.
MODEL_COLORS: dict[str, str] = {
    "Seasonal naive": GREEN,
    "SARIMAX": ORANGE,
    "HistGradientBoosting": BLUE,
}

# Semantic roles in the forecast and anomaly views.
OBSERVED = "#33322e"      # observed demand: a dark neutral, not a categorical hue
FORECAST = BLUE           # the model's point forecast
EXPECTED = ORANGE         # what the model expected, in the anomaly view
INTERVAL_FILL = "rgba(42, 120, 214, 0.15)"   # BLUE at 15% — residual band
ANOMALY = "#c0392b"       # reserved status red; always paired with a marker + label


def style_axes(ax: plt.Axes) -> None:
    """Matplotlib: drop the top/right spines, recede the rest, horizontal grid only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def plotly_layout(**overrides: Any) -> dict[str, Any]:
    """Plotly: a light, low-chrome layout. Pass overrides to merge on top."""
    layout: dict[str, Any] = dict(
        template="plotly_white",
        font=dict(color=INK, size=13),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=56, r=24, t=48, b=48),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=INK),
        ),
        xaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, title_font=dict(color=MUTED)
        ),
        yaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, title_font=dict(color=MUTED)
        ),
    )
    layout.update(overrides)
    return layout
