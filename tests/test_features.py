"""Tests for offer_response.features."""

import pandas as pd
import pytest

from offer_response import cleaning, features
from offer_response.config import FEATURE_COLUMNS


def _master(raw_portfolio, raw_profile, raw_transcript):
    portfolio = cleaning.clean_portfolio(raw_portfolio)
    profile = cleaning.clean_profile(raw_profile)
    transcript = cleaning.parse_transcript(raw_transcript)
    return features.build_master_table(transcript, portfolio, profile)


def test_master_table_keeps_all_transcript_rows(raw_portfolio, raw_profile, raw_transcript):
    master = _master(raw_portfolio, raw_profile, raw_transcript)
    assert len(master) == len(raw_transcript)
    # Transaction rows have no offer: offer columns are NaN.
    txn = master[master["event"] == "transaction"].iloc[0]
    assert pd.isna(txn["offer_type"])
    assert pd.isna(txn["difficulty"])


def test_master_table_renames_rewards(raw_portfolio, raw_profile, raw_transcript):
    master = _master(raw_portfolio, raw_profile, raw_transcript)
    completed = master[master["event"] == "offer completed"].iloc[0]
    assert completed["reward_received"] == 10  # from the transcript event
    assert completed["reward_expected"] == 10  # from the portfolio
    assert "became_member_on" not in master.columns


def test_modeling_frame_excludes_transactions_views_and_informational(
    raw_portfolio, raw_profile, raw_transcript
):
    master = _master(raw_portfolio, raw_profile, raw_transcript)
    X, y, frame = features.build_modeling_frame(master)
    # Fixture: 3 received + 1 completed for bogo/discount; the informational
    # received row, the 2 viewed rows and the 3 transactions are excluded.
    assert len(X) == 4
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == 1
    assert "informational" not in frame.columns


def test_modeling_frame_imputes_missing_demographics(
    raw_portfolio, raw_profile, raw_transcript
):
    master = _master(raw_portfolio, raw_profile, raw_transcript)
    X, y, frame = features.build_modeling_frame(master)
    assert X["income"].isna().sum() == 0
    # cust-3 has unknown demographics: gender imputed to "O" (one row),
    # income imputed to the mean of the known incomes in the frame
    # (112000 + 112000 + 60000) / 3.
    assert int(X["O"].sum()) == 1
    imputed = X.loc[X["O"] == 1, "income"].iloc[0]
    assert imputed == pytest.approx(284000 / 3)
    assert set(X.columns) == set(FEATURE_COLUMNS)


def test_modeling_frame_excludes_amount_and_reward_received(
    raw_portfolio, raw_profile, raw_transcript
):
    """amount is 0 on every offer row; reward_received leaks the target."""
    master = _master(raw_portfolio, raw_profile, raw_transcript)
    X, y, frame = features.build_modeling_frame(master)
    assert "amount" not in X.columns
    assert "reward_received" not in X.columns
    assert "amount" not in FEATURE_COLUMNS


def test_modeling_frame_feature_order(raw_portfolio, raw_profile, raw_transcript):
    master = _master(raw_portfolio, raw_profile, raw_transcript)
    X, y, _frame = features.build_modeling_frame(master)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape[1] == 15
