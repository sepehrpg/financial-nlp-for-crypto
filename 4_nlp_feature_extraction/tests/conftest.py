from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PHASE4_DIR = Path(__file__).resolve().parents[1]
if str(PHASE4_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE4_DIR))


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_id": pd.Series([2, 5, 9, 14], dtype="int64"),
            "text_title_description": [
                "BTC jumps 5.2% to $67,500 after SEC ETF approval!",
                "Is Bitcoin falling?",
                "",
                "Fed signals a rate cut; crypto markets rally.",
            ],
            "Filtered_Text": [
                "btc jump sec etf approval",
                "bitcoin fall",
                None,
                "federal reserve rate cut cryptocurrency rally",
            ],
        }
    )


@pytest.fixture
def phase4_dir() -> Path:
    return PHASE4_DIR
