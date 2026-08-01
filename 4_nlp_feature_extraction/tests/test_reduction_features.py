from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.reduction_features import (
    compile_phase4_method_benchmark,
    run_pca_smoke,
    run_svd_smoke,
    save_reduction_smoke_outputs,
)


def test_pca_smoke_returns_finite_float32_and_preserves_rows() -> None:
    rng = np.random.default_rng(42)
    ids = pd.Series(range(20), name="source_row_id")
    embeddings = {
        "text_title_description__bert": rng.normal(size=(20, 12)).astype(np.float32),
        "Filtered_Text__finbert": rng.normal(size=(20, 12)).astype(np.float32),
    }
    result = run_pca_smoke(
        embeddings,
        source_row_ids=ids,
        settings={
            "n_components": 5,
            "svd_solver": "auto",
            "whiten": False,
            "random_state": 42,
            "output_dtype": "float32",
        },
    )
    assert result["source_row_ids"].tolist() == ids.tolist()
    for matrix in result["reduced"].values():
        assert matrix.shape == (20, 5)
        assert matrix.dtype == np.float32
        assert np.isfinite(matrix).all()
    assert result["benchmark"]["source_row_id_order_preserved"].all()


def test_svd_smoke_keeps_sparse_input_and_returns_finite_float32() -> None:
    rng = np.random.default_rng(7)
    ids = pd.Series(range(25), name="source_row_id")
    dense = rng.random((25, 30), dtype=np.float32)
    dense[dense < 0.85] = 0.0
    matrices = {
        "text_title_description__word_tfidf": sparse.csr_matrix(dense),
        "text_title_description__character_tfidf": sparse.csr_matrix(dense[:, :20]),
    }
    result = run_svd_smoke(
        matrices,
        source_row_ids=ids,
        settings={
            "n_components": 6,
            "algorithm": "randomized",
            "n_iter": 5,
            "random_state": 42,
            "output_dtype": "float32",
        },
        allowed_recipe_names=["word_tfidf", "character_tfidf"],
    )
    assert sparse.isspmatrix_csr(matrices["text_title_description__word_tfidf"])
    for matrix in result["reduced"].values():
        assert matrix.shape == (25, 6)
        assert matrix.dtype == np.float32
        assert np.isfinite(matrix).all()


def test_reduction_outputs_save_with_mappings(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    ids = pd.Series(range(12), name="source_row_id")
    pca = run_pca_smoke(
        {"rep__bert": rng.normal(size=(12, 10)).astype(np.float32)},
        source_row_ids=ids,
        settings={
            "n_components": 4,
            "svd_solver": "auto",
            "whiten": False,
            "random_state": 42,
            "output_dtype": "float32",
        },
    )
    tfidf = sparse.random(12, 15, density=0.2, format="csr", random_state=42)
    svd = run_svd_smoke(
        {"rep__word_tfidf": tfidf},
        source_row_ids=ids,
        settings={
            "n_components": 4,
            "algorithm": "randomized",
            "n_iter": 5,
            "random_state": 42,
            "output_dtype": "float32",
        },
        allowed_recipe_names=["word_tfidf"],
    )
    files = save_reduction_smoke_outputs(
        pca_result=pca,
        svd_result=svd,
        output_directory=tmp_path / "reduction",
        benchmark_path=tmp_path / "reduction_benchmark.csv",
        id_column="source_row_id",
    )
    assert Path(files["pca_array::rep__bert"]).is_file()
    assert Path(files["svd_array::rep__word_tfidf"]).is_file()
    assert pd.read_csv(files["pca_source_row_ids"])["source_row_id"].tolist() == ids.tolist()


def test_final_benchmark_contains_all_method_families(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        '{"feature_rows": 10, "feature_columns": 49, "elapsed_seconds": 1.2, '
        '"feature_validation": {"has_blockers": false}}',
        encoding="utf-8",
    )
    stage2 = tmp_path / "stage2.csv"
    pd.DataFrame(
        [
            {
                "representation": "rep",
                "recipe": "word_tfidf",
                "rows": 10,
                "features": 20,
                "elapsed_seconds": 0.2,
                "approx_sparse_memory_mb": 0.1,
                "finite_values": True,
            }
        ]
    ).to_csv(stage2, index=False)
    transformer = pd.DataFrame(
        [
            {
                "representation": "rep",
                "model": "bert",
                "rows": 10,
                "dimensions": 8,
                "elapsed_seconds": 0.3,
                "approx_embedding_memory_mb": 0.01,
                "finite_values": True,
            }
        ]
    )
    reduction = pd.DataFrame(
        [
            {
                "input_key": "rep__bert",
                "method": "pca",
                "rows": 10,
                "output_dimensions": 4,
                "elapsed_seconds": 0.1,
                "approx_output_memory_mb": 0.001,
                "finite_values": True,
                "fit_scope": "smoke_sample_only",
            }
        ]
    )
    benchmark = compile_phase4_method_benchmark(
        stage1_metadata_path=stage1,
        stage2_benchmark_path=stage2,
        transformer_benchmark=transformer,
        reduction_benchmark=reduction,
    )
    assert set(benchmark["method_family"]) == {
        "row_level_manual",
        "tfidf",
        "frozen_transformer",
        "dimensionality_reduction",
    }
