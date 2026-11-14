"""Data-cleaning steps for the three raw tables.

Each function takes a raw frame and returns a cleaned copy; nothing is
modified in place.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .config import AGE_MISSING_SENTINEL, CHANNELS


def clean_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Clean the offer table.

    - Expands the ``channels`` list into one binary column per channel.
    - Converts offer ``duration`` from days to hours (``duration_h``) so it
      is comparable with transcript times, which are in hours.
    - Renames ``id`` to ``offer_id``.
    """
    df = portfolio.copy()
    for channel in CHANNELS:
        df[channel] = df["channels"].map(
            lambda channels: int(channel in channels)
            if isinstance(channels, (list, tuple, set))
            else 0
        )
    df["duration_h"] = df["duration"] * 24
    df = df.drop(columns=["duration", "channels"])
    df = df.rename(columns={"id": "offer_id"})
    return df


def clean_profile(
    profile: pd.DataFrame,
    reference_date: dt.date | None = None,
) -> pd.DataFrame:
    """Clean the customer table.

    - Renames ``id`` to ``customer_id``.
    - Adds ``m_days``: days between the customer joining (``became_member_on``)
      and ``reference_date``. The reference defaults to the latest join date
      in the data, so the feature is reproducible — the original notebook
      used the current date, which made the feature drift every time the
      notebook was re-run.

    The ``age == 118`` rows are kept as they are: 118 is a sentinel meaning
    "demographics unknown" (those customers have no gender or income), and
    the sentinel itself carries that information into the model. Gender and
    income imputation happens later, in :func:`features.build_modeling_frame`,
    where the imputation values are documented.
    """
    df = profile.copy()
    df = df.rename(columns={"id": "customer_id"})
    join_date = pd.to_datetime(df["became_member_on"].astype(str), format="%Y%m%d")
    if reference_date is None:
        reference_date = join_date.max().date()
    df["m_days"] = (pd.Timestamp(reference_date) - join_date).dt.days
    return df


def parse_transcript(transcript: pd.DataFrame) -> pd.DataFrame:
    """Parse the event transcript into a flat table.

    - Renames ``person`` to ``customer_id`` and ``time`` to ``start_h``
      (hours since the start of the test period).
    - Expands the ``value`` dict column into ``offer_id``, ``amount`` and
      ``reward``. Offer events use either the ``offer id`` or the
      ``offer_id`` key (the raw data is inconsistent); transaction events
      carry ``amount``; completed-offer events carry ``reward``.
    - ``amount`` is kept as a float: the original notebook stored it in an
      integer column, silently truncating e.g. $14.76 to $14.

    The expansion is vectorised; the original notebook parsed the 306k rows
    one by one with ``iterrows``.
    """
    df = transcript.copy()
    df = df.rename(columns={"person": "customer_id", "time": "start_h"})

    values = df["value"].map(lambda v: v if isinstance(v, dict) else {})
    expanded = pd.DataFrame(values.tolist(), index=df.index)

    offer_id = expanded.get("offer id")
    offer_id_underscore = expanded.get("offer_id")
    if offer_id is None:
        offer_id = offer_id_underscore
    elif offer_id_underscore is not None:
        offer_id = offer_id.combine_first(offer_id_underscore)
    df["offer_id"] = offer_id.fillna("N/A") if offer_id is not None else "N/A"

    amount = expanded.get("amount")
    df["amount"] = (
        pd.to_numeric(amount, errors="coerce").fillna(0.0)
        if amount is not None
        else 0.0
    )
    reward = expanded.get("reward")
    df["reward"] = (
        pd.to_numeric(reward, errors="coerce").fillna(0).astype(int)
        if reward is not None
        else 0
    )
    df = df.drop(columns=["value"])
    return df


def count_missing_demographics(profile: pd.DataFrame) -> int:
    """Number of customers whose demographics are unknown (age sentinel)."""
    return int((profile["age"] == AGE_MISSING_SENTINEL).sum())
