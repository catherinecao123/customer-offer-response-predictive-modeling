"""Tests for offer_response.cleaning."""

import datetime as dt

import pandas as pd
import pytest

from offer_response import cleaning
from tests.conftest import BOGO_ID, DISCOUNT_ID


def test_clean_portfolio_channel_dummies(raw_portfolio):
    cleaned = cleaning.clean_portfolio(raw_portfolio)
    bogo = cleaned.loc[cleaned["offer_id"] == BOGO_ID].iloc[0]
    assert (bogo["email"], bogo["mobile"], bogo["social"], bogo["web"]) == (1, 1, 1, 0)
    discount = cleaned.loc[cleaned["offer_id"] == DISCOUNT_ID].iloc[0]
    assert (discount["email"], discount["mobile"], discount["social"], discount["web"]) == (1, 0, 0, 1)


def test_clean_portfolio_duration_converted_to_hours(raw_portfolio):
    cleaned = cleaning.clean_portfolio(raw_portfolio)
    assert "duration" not in cleaned.columns
    assert "channels" not in cleaned.columns
    assert cleaned.loc[cleaned["offer_id"] == BOGO_ID, "duration_h"].iloc[0] == 7 * 24


def test_clean_portfolio_handles_missing_channels():
    frame = pd.DataFrame(
        [{"id": "x", "offer_type": "bogo", "difficulty": 1, "reward": 1,
          "duration": 1, "channels": None}]
    )
    cleaned = cleaning.clean_portfolio(frame)
    assert cleaned[["email", "mobile", "social", "web"]].iloc[0].sum() == 0


def test_clean_portfolio_does_not_mutate_input(raw_portfolio):
    original = raw_portfolio.copy(deep=True)
    cleaning.clean_portfolio(raw_portfolio)
    pd.testing.assert_frame_equal(raw_portfolio, original)


def test_clean_profile_member_days_uses_reference_date(raw_profile):
    cleaned = cleaning.clean_profile(raw_profile, reference_date=dt.date(2018, 7, 26))
    # cust-1 joined 2017-07-15; 2018-07-26 is 376 days later.
    assert cleaned.loc[cleaned["customer_id"] == "cust-1", "m_days"].iloc[0] == 376
    # cust-4 joined on the reference date itself.
    assert cleaned.loc[cleaned["customer_id"] == "cust-4", "m_days"].iloc[0] == 0


def test_clean_profile_defaults_to_latest_join_date(raw_profile):
    cleaned = cleaning.clean_profile(raw_profile)
    # Latest join date in the fixture is 2018-07-26 (cust-4).
    assert cleaned["m_days"].min() == 0
    assert (cleaned["m_days"] >= 0).all()


def test_clean_profile_keeps_age_118_sentinel(raw_profile):
    cleaned = cleaning.clean_profile(raw_profile)
    row = cleaned.loc[cleaned["customer_id"] == "cust-3"].iloc[0]
    assert row["age"] == 118
    assert pd.isna(row["income"])


def test_count_missing_demographics(raw_profile):
    assert cleaning.count_missing_demographics(raw_profile) == 1


def test_parse_transcript_handles_both_offer_id_keys(raw_transcript):
    parsed = cleaning.parse_transcript(raw_transcript)
    received = parsed.loc[parsed["event"] == "offer received", "offer_id"].tolist()
    assert received == [BOGO_ID, DISCOUNT_ID, "offer-info-1", DISCOUNT_ID]
    completed = parsed.loc[parsed["event"] == "offer completed", "offer_id"].iloc[0]
    assert completed == BOGO_ID


def test_parse_transcript_keeps_float_amounts(raw_transcript):
    """Regression test: the original notebook truncated $14.76 to $14."""
    parsed = cleaning.parse_transcript(raw_transcript)
    amounts = parsed.loc[parsed["event"] == "transaction", "amount"].tolist()
    assert amounts == pytest.approx([14.76, 5.5, 3.25])
    assert parsed["amount"].dtype.kind == "f"


def test_parse_transcript_reward_and_na_offer_id(raw_transcript):
    parsed = cleaning.parse_transcript(raw_transcript)
    completed = parsed.loc[parsed["event"] == "offer completed"].iloc[0]
    assert completed["reward"] == 10
    transactions = parsed.loc[parsed["event"] == "transaction"]
    assert (transactions["offer_id"] == "N/A").all()
    assert (transactions["reward"] == 0).all()
    assert "value" not in parsed.columns


def test_parse_transcript_handles_non_dict_values():
    frame = pd.DataFrame(
        [
            {"event": "transaction", "person": "c1", "time": 0, "value": None},
            {"event": "transaction", "person": "c1", "time": 1, "value": "junk"},
        ]
    )
    parsed = cleaning.parse_transcript(frame)
    assert (parsed["offer_id"] == "N/A").all()
    assert (parsed["amount"] == 0.0).all()
