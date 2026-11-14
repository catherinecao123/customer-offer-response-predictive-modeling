"""Descriptive findings: completion rates, view rates, customer segments.

All functions are pure (they only read their inputs) and return plain
Python types so the results can be dumped to JSON.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .cleaning import count_missing_demographics
from .config import (
    EVENT_OFFER_COMPLETED,
    EVENT_OFFER_RECEIVED,
    EVENT_OFFER_VIEWED,
    EVENT_TRANSACTION,
    OFFER_TYPE_INFORMATIONAL,
)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(
    master: pd.DataFrame,
    portfolio: pd.DataFrame,
    profile: pd.DataFrame,
    transcript: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute the headline findings for the report and README."""
    events = transcript["event"].value_counts()
    # Completion stats cover bogo/discount only: informational offers are
    # ads with no completion event, so including them would understate the
    # completion rate.
    offer_rows = master[
        master["event"].isin([EVENT_OFFER_RECEIVED, EVENT_OFFER_COMPLETED])
        & (master["offer_type"] != OFFER_TYPE_INFORMATIONAL)
    ]
    received = offer_rows[offer_rows["event"] == EVENT_OFFER_RECEIVED]
    completed = offer_rows[offer_rows["event"] == EVENT_OFFER_COMPLETED]

    completion_by_type = {}
    for offer_type in sorted(received["offer_type"].dropna().unique()):
        n_received = int((received["offer_type"] == offer_type).sum())
        n_completed = int((completed["offer_type"] == offer_type).sum())
        completion_by_type[offer_type] = {
            "received": n_received,
            "completed": n_completed,
            "completion_rate": _rate(n_completed, n_received),
        }

    informational = master[master["offer_type"] == OFFER_TYPE_INFORMATIONAL]
    info_received = int((informational["event"] == EVENT_OFFER_RECEIVED).sum())
    info_viewed = int((informational["event"] == EVENT_OFFER_VIEWED).sum())

    known_gender = profile[profile["gender"].notna()]
    gender_share = {
        gender: round(float(share), 4)
        for gender, share in known_gender["gender"].value_counts(normalize=True).items()
    }

    return {
        "dataset": {
            "n_offers": int(len(portfolio)),
            "offer_types": sorted(portfolio["offer_type"].unique().tolist()),
            "n_customers": int(profile["customer_id"].nunique()),
            "n_events": int(len(transcript)),
            "events_by_type": {k: int(v) for k, v in events.items()},
            "customers_missing_demographics": count_missing_demographics(profile),
        },
        "demographics": {
            "gender_share_known": gender_share,
            "mean_income_known": round(float(known_gender["income"].mean()), 2),
            "mean_age_known": round(float(known_gender["age"].mean()), 2),
        },
        "offer_completion": {
            "received_bogo_discount": int(len(received)),
            "completed": int(len(completed)),
            "completion_rate": _rate(len(completed), len(received)),
            "by_offer_type": completion_by_type,
        },
        "informational_offers": {
            "received": info_received,
            "viewed": info_viewed,
            "view_rate": _rate(info_viewed, info_received),
        },
        "customer_segments": customer_segments(transcript),
    }


def customer_segments(transcript: pd.DataFrame) -> Dict[str, int]:
    """Split customers by offer activity and purchasing.

    The four segments partition all customers seen in the transcript:

    - ``completed_and_transacted``: completed at least one offer and made
      at least one purchase.
    - ``transacted_never_completed``: made a purchase, never completed an
      offer.
    - ``offer_activity_no_transaction``: received/viewed offers but never
      purchased.
    - ``transacted_no_offer``: purchased without ever receiving an offer.

    Implemented with customer-id sets, not with the row-level self-join
    the original notebook used (which produced a ~1.5M-row intermediate).
    """
    transacted = set(
        transcript.loc[transcript["event"] == EVENT_TRANSACTION, "customer_id"]
    )
    completed = set(
        transcript.loc[transcript["event"] == EVENT_OFFER_COMPLETED, "customer_id"]
    )
    offer_activity = set(
        transcript.loc[
            transcript["event"].isin(
                [EVENT_OFFER_RECEIVED, EVENT_OFFER_VIEWED, EVENT_OFFER_COMPLETED]
            ),
            "customer_id",
        ]
    )
    all_customers = set(transcript["customer_id"])

    completed_and_transacted = completed & transacted
    transacted_never_completed = transacted - completed
    offer_activity_no_transaction = offer_activity - transacted
    transacted_no_offer = transacted - offer_activity
    return {
        "completed_and_transacted": len(completed_and_transacted),
        "transacted_never_completed": len(transacted_never_completed),
        "offer_activity_no_transaction": len(offer_activity_no_transaction),
        "transacted_no_offer": len(transacted_no_offer),
        "total_customers_in_transcript": len(all_customers),
    }
