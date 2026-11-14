"""Shared constants for the offer-response pipeline."""

from pathlib import Path

# Repository root (two levels above this file: src/offer_response/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_SOURCE = REPO_ROOT / "data.zip"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

# Reproducibility.
RANDOM_STATE = 4
TEST_SIZE = 0.2

# In the profile data, age 118 is a sentinel for "demographics unknown":
# every customer with age 118 has missing gender and income (2,175 of 17,000).
AGE_MISSING_SENTINEL = 118

# Marketing channels an offer can be sent through (portfolio `channels`).
CHANNELS = ["email", "mobile", "social", "web"]

# Events recorded in the transcript.
EVENT_TRANSACTION = "transaction"
EVENT_OFFER_RECEIVED = "offer received"
EVENT_OFFER_VIEWED = "offer viewed"
EVENT_OFFER_COMPLETED = "offer completed"

# Offer types. Informational offers are ads: they have no completion
# event, so they are excluded from the completion-prediction task.
OFFER_TYPE_INFORMATIONAL = "informational"

# Features used by the models, in order. Notes on what is NOT here:
# - `amount` is excluded: it is only recorded on transaction events, so it
#   is 0 for every row in the modeling frame (offer received/completed).
# - `reward_received` is excluded: it is only nonzero on completed rows,
#   so it leaks the target.
# - `start_h` (event hour) and the raw ids carry no predictive signal for
#   this task and are excluded.
FEATURE_COLUMNS = [
    "age",
    "income",
    "difficulty",
    "m_days",
    "duration_h",
    "reward_expected",
    "email",
    "mobile",
    "social",
    "web",
    "bogo",
    "discount",
    "F",
    "M",
    "O",
]

TARGET_COLUMN = "completed"
