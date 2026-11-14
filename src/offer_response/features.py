"""Build the master table and the modeling frame."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import (
    EVENT_OFFER_COMPLETED,
    EVENT_OFFER_RECEIVED,
    FEATURE_COLUMNS,
    OFFER_TYPE_INFORMATIONAL,
    TARGET_COLUMN,
)

MASTER_COLUMNS = [
    "customer_id",
    "offer_id",
    "event",
    "gender",
    "age",
    "income",
    "offer_type",
    "difficulty",
    "m_days",
    "start_h",
    "duration_h",
    "reward_expected",
    "reward_received",
    "amount",
    "email",
    "mobile",
    "social",
    "web",
]


def build_master_table(
    transcript: pd.DataFrame,
    portfolio: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join the transcript to the offer and customer tables.

    Transaction rows have no offer, so their offer columns are NaN; every
    transcript row has a customer, so the customer columns are filled.
    ``reward_x`` (reward actually received, from the transcript) and
    ``reward_y`` (reward the offer promises, from the portfolio) are renamed
    to ``reward_received`` / ``reward_expected``.
    """
    merged = transcript.merge(portfolio, how="left", on="offer_id")
    merged = merged.merge(profile, how="left", on="customer_id")
    merged = merged.drop(columns=["became_member_on"])
    merged = merged.rename(
        columns={"reward_x": "reward_received", "reward_y": "reward_expected"}
    )
    return merged[[c for c in MASTER_COLUMNS if c in merged.columns]]


def build_modeling_frame(
    master: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the supervised-learning frame from the master table.

    - Keeps only ``offer received`` (label 0) and ``offer completed``
      (label 1) rows; transaction and ``offer viewed`` rows are not
      completion outcomes.
    - Drops informational offers: they are ads with no completion event,
      so they cannot be labelled.
    - Imputes missing gender with ``"O"`` (already a category in the data)
      and missing income with the frame's mean income.
    - One-hot encodes offer type, gender and the event (target) column.

    Returns ``(X, y, frame)`` where ``frame`` is the encoded modeling frame
    (useful for inspection) and ``X`` holds :data:`config.FEATURE_COLUMNS`.

    Known limitation: an offer that is completed appears twice — once as a
    received row (label 0) and once as a completed row (label 1) — with
    identical features, because the features describe the customer and the
    offer, not the individual offer instance. The two classes are therefore
    not separable from these features alone, which caps the achievable
    score. See the README for details.
    """
    df = master[
        master["event"].isin([EVENT_OFFER_RECEIVED, EVENT_OFFER_COMPLETED])
    ].copy()
    df = df[df["offer_type"] != OFFER_TYPE_INFORMATIONAL].copy()

    df["gender"] = df["gender"].fillna("O")
    df["income"] = df["income"].fillna(df["income"].mean())

    df = pd.get_dummies(
        df, columns=["offer_type", "gender", "event"], dtype=int
    )
    df = df.drop(
        columns=[
            "customer_id",
            "offer_id",
            "start_h",
            "reward_received",
            "event_offer received",
        ]
    )
    df = df.rename(
        columns={
            "offer_type_bogo": "bogo",
            "offer_type_discount": "discount",
            "gender_F": "F",
            "gender_M": "M",
            "gender_O": "O",
            "event_offer completed": TARGET_COLUMN,
        }
    )
    # A one-hot column is missing entirely when its category never occurs
    # (e.g. no "O" customers in a small sample); fill it with 0 so the
    # feature set is always complete.
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y, df
