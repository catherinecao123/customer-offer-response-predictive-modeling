"""Tests for offer_response.analysis."""

import pytest

from offer_response import analysis, cleaning, features


def _tables(raw_portfolio, raw_profile, raw_transcript):
    portfolio = cleaning.clean_portfolio(raw_portfolio)
    profile = cleaning.clean_profile(raw_profile)
    transcript = cleaning.parse_transcript(raw_transcript)
    master = features.build_master_table(transcript, portfolio, profile)
    return master, portfolio, profile, transcript


def test_summarize_counts(raw_portfolio, raw_profile, raw_transcript):
    master, portfolio, profile, transcript = _tables(
        raw_portfolio, raw_profile, raw_transcript
    )
    report = analysis.summarize(master, portfolio, profile, transcript)
    assert report["dataset"]["n_offers"] == 3
    assert report["dataset"]["n_customers"] == 4
    assert report["dataset"]["n_events"] == 10
    assert report["dataset"]["customers_missing_demographics"] == 1
    assert report["dataset"]["events_by_type"]["offer completed"] == 1


def test_summarize_completion_rate(raw_portfolio, raw_profile, raw_transcript):
    master, portfolio, profile, transcript = _tables(
        raw_portfolio, raw_profile, raw_transcript
    )
    report = analysis.summarize(master, portfolio, profile, transcript)
    # bogo/discount: 3 received, 1 completed.
    assert report["offer_completion"]["received_bogo_discount"] == 3
    assert report["offer_completion"]["completed"] == 1
    assert report["offer_completion"]["completion_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["offer_completion"]["by_offer_type"]["bogo"]["completion_rate"] == 1.0
    assert report["offer_completion"]["by_offer_type"]["discount"]["completion_rate"] == 0.0


def test_summarize_informational_view_rate(raw_portfolio, raw_profile, raw_transcript):
    master, portfolio, profile, transcript = _tables(
        raw_portfolio, raw_profile, raw_transcript
    )
    report = analysis.summarize(master, portfolio, profile, transcript)
    assert report["informational_offers"] == {
        "received": 1, "viewed": 1, "view_rate": 1.0,
    }


def test_customer_segments_partition_customers(raw_portfolio, raw_profile, raw_transcript):
    _master, _portfolio, _profile, transcript = _tables(
        raw_portfolio, raw_profile, raw_transcript
    )
    segments = analysis.customer_segments(transcript)
    # cust-1 completed + transacted; cust-4 transacted, no offer;
    # cust-2/cust-3 had offer activity but no transaction; cust-5
    # transacted with no offer.
    assert segments["completed_and_transacted"] == 1
    assert segments["transacted_never_completed"] == 2  # cust-4 and cust-5
    assert segments["offer_activity_no_transaction"] == 2  # cust-2, cust-3
    assert segments["transacted_no_offer"] == 2  # cust-4 and cust-5
    assert segments["total_customers_in_transcript"] == 5


def test_customer_segments_sum_to_total(raw_portfolio, raw_profile, raw_transcript):
    _master, _portfolio, _profile, transcript = _tables(
        raw_portfolio, raw_profile, raw_transcript
    )
    segments = analysis.customer_segments(transcript)
    # The first three segments partition all customers; transacted_no_offer
    # is a subset of transacted_never_completed, so it is not added again.
    total = (
        segments["completed_and_transacted"]
        + segments["transacted_never_completed"]
        + segments["offer_activity_no_transaction"]
    )
    assert total == segments["total_customers_in_transcript"]
