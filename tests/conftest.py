"""Small synthetic fixtures mirroring the real data's shape and quirks."""

import pandas as pd
import pytest

# Offer ids used across fixtures (kept short for readability).
BOGO_ID = "offer-bogo-1"
DISCOUNT_ID = "offer-discount-1"
INFO_ID = "offer-info-1"


@pytest.fixture
def raw_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": BOGO_ID,
                "offer_type": "bogo",
                "difficulty": 10,
                "reward": 10,
                "duration": 7,
                "channels": ["email", "mobile", "social"],
            },
            {
                "id": DISCOUNT_ID,
                "offer_type": "discount",
                "difficulty": 5,
                "reward": 2,
                "duration": 5,
                "channels": ["web", "email"],
            },
            {
                "id": INFO_ID,
                "offer_type": "informational",
                "difficulty": 0,
                "reward": 0,
                "duration": 3,
                "channels": ["email"],
            },
        ]
    )


@pytest.fixture
def raw_profile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Known demographics.
            {"id": "cust-1", "age": 55, "became_member_on": 20170715,
             "gender": "F", "income": 112000.0},
            {"id": "cust-2", "age": 40, "became_member_on": 20180101,
             "gender": "M", "income": 60000.0},
            # Unknown demographics: age 118 sentinel, missing gender/income.
            {"id": "cust-3", "age": 118, "became_member_on": 20170212,
             "gender": None, "income": float("nan")},
            {"id": "cust-4", "age": 30, "became_member_on": 20180726,
             "gender": "O", "income": 45000.0},
        ]
    )


@pytest.fixture
def raw_transcript() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event": "offer received", "person": "cust-1", "time": 0,
             "value": {"offer id": BOGO_ID}},
            # Completed events use the "offer_id" key (raw data is inconsistent).
            {"event": "offer completed", "person": "cust-1", "time": 24,
             "value": {"offer_id": BOGO_ID, "reward": 10}},
            {"event": "offer received", "person": "cust-2", "time": 0,
             "value": {"offer id": DISCOUNT_ID}},
            {"event": "offer viewed", "person": "cust-2", "time": 6,
             "value": {"offer id": DISCOUNT_ID}},
            {"event": "offer received", "person": "cust-3", "time": 0,
             "value": {"offer id": INFO_ID}},
            {"event": "offer viewed", "person": "cust-3", "time": 6,
             "value": {"offer id": INFO_ID}},
            # cust-3 (unknown demographics) also received a discount offer,
            # so the modeling frame exercises gender/income imputation.
            {"event": "offer received", "person": "cust-3", "time": 48,
             "value": {"offer id": DISCOUNT_ID}},
            # Transactions carry a float amount — must not be truncated.
            {"event": "transaction", "person": "cust-1", "time": 12,
             "value": {"amount": 14.76}},
            {"event": "transaction", "person": "cust-4", "time": 12,
             "value": {"amount": 5.5}},
            # A transaction from a customer with no offer activity at all.
            {"event": "transaction", "person": "cust-5", "time": 12,
             "value": {"amount": 3.25}},
        ]
    )
