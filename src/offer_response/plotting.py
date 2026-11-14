"""Figures for the report. Each function saves a PNG and returns its path.

Uses the non-interactive Agg backend so the pipeline runs headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import EVENT_OFFER_COMPLETED, EVENT_OFFER_RECEIVED


def plot_event_counts(transcript: pd.DataFrame, path: str | Path) -> Path:
    counts = transcript["event"].value_counts()
    order = [
        "transaction",
        "offer received",
        "offer viewed",
        "offer completed",
    ]
    counts = counts.reindex([e for e in order if e in counts.index])
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette()[0])
    ax.bar_label(bars, fmt="{:,}", padding=3)
    ax.set_xlabel("Event type")
    ax.set_ylabel("# of events")
    ax.set_title("Events in the transcript (306,534 total)")
    plt.xticks(rotation=15)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_completion_by_offer_type(master: pd.DataFrame, path: str | Path) -> Path:
    df = master[master["event"].isin([EVENT_OFFER_RECEIVED, EVENT_OFFER_COMPLETED])]
    df = df[df["offer_type"].isin(["bogo", "discount"])]
    counts = (
        df.groupby(["offer_type", "event"])
        .size()
        .rename("count")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="offer_type", y="count", hue="event", ax=ax)
    ax.set_xlabel("Offer type")
    ax.set_ylabel("# of offers")
    ax.set_title("Offers received vs. completed, by offer type")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_income_by_gender(profile: pd.DataFrame, path: str | Path) -> Path:
    """Income distributions per gender (customers with known demographics).

    Uses ``kdeplot`` — the original notebook used ``seaborn.distplot``,
    which was removed in seaborn 0.12.
    """
    df = profile[profile["gender"].notna() & profile["income"].notna()]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df, x="income", hue="gender", fill=False, ax=ax)
    ax.set_xlabel("Income ($)")
    ax.set_ylabel("Density")
    ax.set_title("Income distribution by gender")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
