from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_cumulative_pnl(
    unhedged_pnl: pd.Series,
    hedged_pnl: pd.Series,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cumulative_unhedged = (1.0 + unhedged_pnl).cumprod()
    cumulative_hedged = (1.0 + hedged_pnl).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative_unhedged.index, cumulative_unhedged.values, label="Unhedged")
    ax.plot(cumulative_hedged.index, cumulative_hedged.values, label="Hedged")
    ax.set_title("Cumulative P&L Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pnl_histogram(
    unhedged_pnl: pd.Series,
    hedged_pnl: pd.Series,
    output_path: Path,
    bins: int = 50,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(unhedged_pnl, bins=bins, alpha=0.55, label="Unhedged", density=True)
    ax.hist(hedged_pnl, bins=bins, alpha=0.55, label="Hedged", density=True)
    ax.set_title("P&L Distribution")
    ax.set_xlabel("P&L")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rolling_hedge_effectiveness(
    unhedged_pnl: pd.Series,
    hedged_pnl: pd.Series,
    output_path: Path,
    window: int = 60,
) -> None:
    if window < 2:
        raise ValueError("window must be at least 2.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_index = unhedged_pnl.index.intersection(hedged_pnl.index).sort_values()
    u = unhedged_pnl.loc[aligned_index]
    h = hedged_pnl.loc[aligned_index]

    rolling_unhedged_var = u.rolling(window=window).var(ddof=1)
    rolling_hedged_var = h.rolling(window=window).var(ddof=1)
    effectiveness = 1.0 - (rolling_hedged_var / rolling_unhedged_var)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(effectiveness.index, effectiveness.values, label=f"Window={window}")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title("Rolling Hedge Effectiveness")
    ax.set_xlabel("Date")
    ax.set_ylabel("1 - Var(Hedged)/Var(Unhedged)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_hedge_ratios(
    hedge_ratios: pd.Series | pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ratio_frame = (
        hedge_ratios.to_frame(name=hedge_ratios.name or "hedge_ratio")
        if isinstance(hedge_ratios, pd.Series)
        else hedge_ratios.copy()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for column in ratio_frame.columns:
        ax.plot(ratio_frame.index, ratio_frame[column], label=str(column))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title("Hedge Ratios Through Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Beta")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cumulative_returns(
    unhedged_returns: pd.Series,
    hedged_returns: pd.Series,
    output_path: Path,
) -> None:
    """Backward-compatible alias."""
    plot_cumulative_pnl(
        unhedged_pnl=unhedged_returns,
        hedged_pnl=hedged_returns,
        output_path=output_path,
    )
