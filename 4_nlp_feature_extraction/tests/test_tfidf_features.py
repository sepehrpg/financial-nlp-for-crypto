from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.tfidf_features import (
    load_feature_recipes,
    make_smoke_sample,
    run_tfidf_smoke_benchmark,
    save_tfidf_smoke_outputs,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source_row_id": [9, 12, 18, 25, 31, 44, 50, 61],
        "text_title_description": [
            "Bitcoin ETF approval lifts market sentiment",
            "",
            "BTC drops 5 percent after regulatory news",
            "Federal Reserve rate cut supports risk assets",
            "Exchange suffers security hack and liquidation",
            "Bitcoin rally continues as ETF demand rises",
            "SEC lawsuit pressures crypto markets",
            "Options traders expect higher volatility",
        ],
        "Filtered_Text": [
            "bitcoin etf approval lifts market sentiment",
            "short text",
            "btc drops percent regulatory news",
            "federal reserve rate cut supports risk assets",
            "exchange security hack liquidation",
            "bitcoin rally etf demand rises",
            "sec lawsuit pressures crypto markets",
            "",
        ],
    })


def test_sample_is_reproducible_and_preserves_source_order() -> None:
    frame = pd.concat([_frame()] * 5, ignore_index=True)
    frame["source_row_id"] = range(100, 100 + len(frame))
    first = make_smoke_sample(
        frame,
        id_column="source_row_id",
        representations=["text_title_description", "Filtered_Text"],
        sample_size=20,
        random_state=42,
        include_all_empty_rows=True,
    )
    second = make_smoke_sample(
        frame,
        id_column="source_row_id",
        representations=["text_title_description", "Filtered_Text"],
        sample_size=20,
        random_state=42,
        include_all_empty_rows=True,
    )
    assert first["source_row_id"].tolist() == second["source_row_id"].tolist()
    assert first["source_row_id"].is_monotonic_increasing
    assert first["text_title_description"].eq("").any()
    assert first["Filtered_Text"].eq("").any()


def test_all_recipes_return_sparse_finite_matrices_in_id_order(phase4_dir: Path) -> None:
    config = load_feature_recipes(phase4_dir / "configs" / "feature_recipes.yaml")
    # Lower thresholds for the tiny fixture while preserving the three recipe types.
    for recipe in ("word_tfidf", "character_tfidf"):
        config["recipes"][recipe]["vectorizer"]["min_df"] = 1
        config["recipes"][recipe]["vectorizer"]["max_df"] = 1.0
        config["recipes"][recipe]["vectorizer"]["max_features"] = 200

    frame = _frame()
    result = run_tfidf_smoke_benchmark(
        frame,
        id_column="source_row_id",
        representations=config["input"]["representations"],
        recipes=config["recipes"],
    )

    assert result["source_row_ids"].tolist() == frame["source_row_id"].tolist()
    assert len(result["matrices"]) == 6
    for matrix in result["matrices"].values():
        assert sparse.isspmatrix_csr(matrix)
        assert matrix.shape[0] == len(frame)
        assert np.isfinite(matrix.data).all()

    combined = result["matrices"]["text_title_description__word_character_tfidf"]
    word = result["matrices"]["text_title_description__word_tfidf"]
    char = result["matrices"]["text_title_description__character_tfidf"]
    assert combined.shape[1] == word.shape[1] + char.shape[1]
    assert result["benchmark"]["source_row_id_order_preserved"].all()
    assert result["benchmark"]["is_sparse_csr"].all()
    assert result["benchmark"]["finite_values"].all()


def test_sparse_outputs_can_be_saved_and_reloaded(tmp_path: Path, phase4_dir: Path) -> None:
    config = load_feature_recipes(phase4_dir / "configs" / "feature_recipes.yaml")
    for recipe in ("word_tfidf", "character_tfidf"):
        config["recipes"][recipe]["vectorizer"]["min_df"] = 1
        config["recipes"][recipe]["vectorizer"]["max_df"] = 1.0
        config["recipes"][recipe]["vectorizer"]["max_features"] = 100

    result = run_tfidf_smoke_benchmark(
        _frame(),
        id_column="source_row_id",
        representations=config["input"]["representations"],
        recipes=config["recipes"],
    )
    files = save_tfidf_smoke_outputs(
        result,
        output_directory=tmp_path / "matrices",
        benchmark_path=tmp_path / "benchmark.csv",
        metadata_path=tmp_path / "metadata.json",
        config=config,
    )
    loaded = sparse.load_npz(files["matrix::text_title_description__word_tfidf"])
    assert sparse.issparse(loaded)
    saved_ids = pd.read_csv(files["source_row_ids"])["source_row_id"].tolist()
    assert saved_ids == _frame()["source_row_id"].tolist()
