import numpy as np
import pandas as pd

from src.row_features import build_row_level_features, extract_representation_features


KEYWORDS = {
    "crypto_assets": ["bitcoin", "btc", "crypto"],
    "regulation": ["sec", "approval"],
    "instruments": ["etf"],
    "macro": ["federal reserve", "fed", "rate cut"],
}


def test_explainable_counts_are_correct() -> None:
    features = extract_representation_features(
        pd.Series(["BTC jumps 5.2% to $67,500 after SEC ETF approval!"]),
        prefix="text",
        currency_symbols=["$", "€", "₿"],
        keyword_groups=KEYWORDS,
    )
    row = features.iloc[0]
    assert row["text__number_count"] == 2
    assert row["text__percentage_count"] == 1
    assert row["text__currency_symbol_count"] == 1
    assert row["text__exclamation_mark_count"] == 1
    assert row["text__keyword_crypto_assets_count"] == 1
    assert row["text__keyword_regulation_count"] == 2
    assert row["text__keyword_instruments_count"] == 1


def test_build_features_preserves_id_order_and_numeric_types(
    sample_frame: pd.DataFrame,
) -> None:
    result = build_row_level_features(
        sample_frame,
        id_column="source_row_id",
        representations=["text_title_description", "Filtered_Text"],
        representation_prefixes={
            "text_title_description": "title_description",
            "Filtered_Text": "filtered_text",
        },
        currency_symbols=["$", "€", "₿"],
        keyword_groups=KEYWORDS,
    )
    assert result["source_row_id"].tolist() == [2, 5, 9, 14]
    assert result.shape[0] == sample_frame.shape[0]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes)
    assert np.isfinite(result.drop(columns="source_row_id").to_numpy()).all()
    assert not result.isna().any().any()
