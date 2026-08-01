"""PCA/SVD smoke-test helpers and the final Phase 4 benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD


def _safe_component_count(requested: int, rows: int, columns: int) -> int:
    """Choose a valid smoke-test component count and keep at least one component."""

    upper = min(rows - 1, columns - 1)
    if upper < 1:
        raise ValueError(
            "Dimensionality reduction needs at least two rows and two input features."
        )
    return min(int(requested), int(upper))


def run_pca_smoke(
    embeddings: Mapping[str, np.ndarray],
    *,
    source_row_ids: pd.Series,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit PCA only on dense smoke embeddings and return float32 arrays."""

    reduced: dict[str, np.ndarray] = {}
    estimators: dict[str, PCA] = {}
    rows: list[dict[str, Any]] = []

    for key, matrix in embeddings.items():
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ValueError(f"PCA input {key!r} must be a 2D dense array.")
        if sparse.issparse(matrix):
            raise TypeError("PCA smoke input must be dense; use TruncatedSVD for sparse TF-IDF.")
        if dense.shape[0] != len(source_row_ids):
            raise RuntimeError(f"PCA input {key!r} does not match the ID mapping.")
        if not np.isfinite(dense).all():
            raise RuntimeError(f"PCA input {key!r} contains NaN or infinity.")

        n_components = _safe_component_count(
            int(settings["n_components"]), dense.shape[0], dense.shape[1]
        )
        estimator = PCA(
            n_components=n_components,
            svd_solver=str(settings.get("svd_solver", "auto")),
            whiten=bool(settings.get("whiten", False)),
            random_state=int(settings.get("random_state", 42)),
        )
        started = perf_counter()
        output = estimator.fit_transform(dense).astype(np.float32, copy=False)
        elapsed = perf_counter() - started
        if output.shape != (dense.shape[0], n_components):
            raise RuntimeError("PCA changed row count or returned an unexpected dimension.")
        if not np.isfinite(output).all():
            raise RuntimeError("PCA produced NaN or infinite values.")

        reduced[key] = output
        estimators[key] = estimator
        rows.append(
            {
                "input_key": key,
                "method": "pca",
                "rows": int(output.shape[0]),
                "input_dimensions": int(dense.shape[1]),
                "output_dimensions": int(output.shape[1]),
                "shape": f"({output.shape[0]}, {output.shape[1]})",
                "dtype": str(output.dtype),
                "elapsed_seconds": float(elapsed),
                "approx_output_memory_mb": float(output.nbytes / (1024**2)),
                "explained_variance_ratio_sum": float(
                    estimator.explained_variance_ratio_.sum()
                ),
                "finite_values": bool(np.isfinite(output).all()),
                "source_row_id_order_preserved": True,
                "fit_scope": "smoke_sample_only",
            }
        )

    return {
        "source_row_ids": source_row_ids.reset_index(drop=True).copy(),
        "reduced": reduced,
        "estimators": estimators,
        "benchmark": pd.DataFrame(rows),
    }


def run_svd_smoke(
    matrices: Mapping[str, sparse.spmatrix],
    *,
    source_row_ids: pd.Series,
    settings: Mapping[str, Any],
    allowed_recipe_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit TruncatedSVD on sparse TF-IDF smoke matrices without densifying inputs."""

    reduced: dict[str, np.ndarray] = {}
    estimators: dict[str, TruncatedSVD] = {}
    rows: list[dict[str, Any]] = []
    allowed = set(allowed_recipe_names or [])

    for key, matrix in matrices.items():
        recipe_name = key.split("__", maxsplit=1)[1] if "__" in key else key
        if allowed and recipe_name not in allowed:
            continue
        if not sparse.issparse(matrix):
            raise TypeError(f"TruncatedSVD input {key!r} must remain sparse.")
        csr = matrix.tocsr()
        if csr.shape[0] != len(source_row_ids):
            raise RuntimeError(f"SVD input {key!r} does not match the ID mapping.")
        if not np.isfinite(csr.data).all():
            raise RuntimeError(f"SVD input {key!r} contains NaN or infinity.")

        n_components = _safe_component_count(
            int(settings["n_components"]), csr.shape[0], csr.shape[1]
        )
        estimator = TruncatedSVD(
            n_components=n_components,
            algorithm=str(settings.get("algorithm", "randomized")),
            n_iter=int(settings.get("n_iter", 7)),
            random_state=int(settings.get("random_state", 42)),
        )
        started = perf_counter()
        output = estimator.fit_transform(csr).astype(np.float32, copy=False)
        elapsed = perf_counter() - started
        if output.shape != (csr.shape[0], n_components):
            raise RuntimeError("TruncatedSVD changed row count or output dimension.")
        if not np.isfinite(output).all():
            raise RuntimeError("TruncatedSVD produced NaN or infinite values.")

        reduced[key] = output
        estimators[key] = estimator
        rows.append(
            {
                "input_key": key,
                "method": "truncated_svd",
                "rows": int(output.shape[0]),
                "input_dimensions": int(csr.shape[1]),
                "output_dimensions": int(output.shape[1]),
                "shape": f"({output.shape[0]}, {output.shape[1]})",
                "dtype": str(output.dtype),
                "elapsed_seconds": float(elapsed),
                "approx_output_memory_mb": float(output.nbytes / (1024**2)),
                "explained_variance_ratio_sum": float(
                    estimator.explained_variance_ratio_.sum()
                ),
                "finite_values": bool(np.isfinite(output).all()),
                "source_row_id_order_preserved": True,
                "fit_scope": "smoke_sample_only",
            }
        )

    if not rows:
        raise ValueError("No sparse TF-IDF matrices matched the requested SVD recipes.")
    return {
        "source_row_ids": source_row_ids.reset_index(drop=True).copy(),
        "reduced": reduced,
        "estimators": estimators,
        "benchmark": pd.DataFrame(rows),
    }


def save_reduction_smoke_outputs(
    *,
    pca_result: Mapping[str, Any],
    svd_result: Mapping[str, Any],
    output_directory: Path,
    benchmark_path: Path,
    id_column: str,
) -> dict[str, str]:
    """Save small dense reduction arrays, estimators, mappings, and benchmark."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    benchmark_path = Path(benchmark_path)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    pca_ids_path = output_directory / "pca_smoke_source_row_ids.csv"
    pd.DataFrame({id_column: pca_result["source_row_ids"]}).to_csv(
        pca_ids_path, index=False
    )
    outputs["pca_source_row_ids"] = str(pca_ids_path)
    for key, matrix in pca_result["reduced"].items():
        array_path = output_directory / f"{key}_pca.npy"
        estimator_path = output_directory / f"{key}_pca.joblib"
        np.save(array_path, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
        joblib.dump(pca_result["estimators"][key], estimator_path)
        outputs[f"pca_array::{key}"] = str(array_path)
        outputs[f"pca_estimator::{key}"] = str(estimator_path)

    svd_ids_path = output_directory / "svd_smoke_source_row_ids.csv"
    pd.DataFrame({id_column: svd_result["source_row_ids"]}).to_csv(
        svd_ids_path, index=False
    )
    outputs["svd_source_row_ids"] = str(svd_ids_path)
    for key, matrix in svd_result["reduced"].items():
        array_path = output_directory / f"{key}_svd.npy"
        estimator_path = output_directory / f"{key}_svd.joblib"
        np.save(array_path, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
        joblib.dump(svd_result["estimators"][key], estimator_path)
        outputs[f"svd_array::{key}"] = str(array_path)
        outputs[f"svd_estimator::{key}"] = str(estimator_path)

    benchmark = pd.concat(
        [pca_result["benchmark"], svd_result["benchmark"]], ignore_index=True
    )
    benchmark.to_csv(benchmark_path, index=False)
    outputs["benchmark"] = str(benchmark_path)
    return outputs


def compile_phase4_method_benchmark(
    *,
    stage1_metadata_path: Path | None,
    stage2_benchmark_path: Path | None,
    transformer_benchmark: pd.DataFrame,
    reduction_benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Create one comparable inventory of all Phase 4 feature families.

    Runtime and memory are descriptive because smoke samples and hardware differ.
    Semantic predictive quality is deliberately deferred to Phase 7 evaluation.
    """

    rows: list[dict[str, Any]] = []
    if stage1_metadata_path is not None and Path(stage1_metadata_path).is_file():
        metadata = json.loads(Path(stage1_metadata_path).read_text(encoding="utf-8"))
        rows.append(
            {
                "stage": "04_01",
                "method_family": "row_level_manual",
                "representation": "both representations",
                "method": "explainable_row_features",
                "rows": metadata.get("feature_rows"),
                "dimensions": max(int(metadata.get("feature_columns", 1)) - 1, 0),
                "storage": "dense_parquet",
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "memory_mb": None,
                "finite_values": not bool(
                    metadata.get("feature_validation", {}).get("has_blockers", False)
                ),
                "fit_scope": "row_local_full_dataset",
                "semantic_quality_status": "not_evaluated_in_phase4",
            }
        )

    if stage2_benchmark_path is not None and Path(stage2_benchmark_path).is_file():
        stage2 = pd.read_csv(stage2_benchmark_path)
        for record in stage2.to_dict(orient="records"):
            rows.append(
                {
                    "stage": "04_02",
                    "method_family": "tfidf",
                    "representation": record.get("representation"),
                    "method": record.get("recipe"),
                    "rows": record.get("rows"),
                    "dimensions": record.get("features"),
                    "storage": "csr_sparse_npz",
                    "elapsed_seconds": record.get("elapsed_seconds"),
                    "memory_mb": record.get("approx_sparse_memory_mb"),
                    "finite_values": record.get("finite_values"),
                    "fit_scope": "smoke_sample_only",
                    "semantic_quality_status": "not_evaluated_in_phase4",
                }
            )

    for record in transformer_benchmark.to_dict(orient="records"):
        rows.append(
            {
                "stage": "04_03",
                "method_family": "frozen_transformer",
                "representation": record.get("representation"),
                "method": record.get("model"),
                "rows": record.get("rows"),
                "dimensions": record.get("dimensions"),
                "storage": "dense_float32_npy",
                "elapsed_seconds": record.get("elapsed_seconds"),
                "memory_mb": record.get("approx_embedding_memory_mb"),
                "finite_values": record.get("finite_values"),
                "fit_scope": "frozen_external_model",
                "semantic_quality_status": "not_evaluated_in_phase4",
            }
        )

    for record in reduction_benchmark.to_dict(orient="records"):
        rows.append(
            {
                "stage": "04_03",
                "method_family": "dimensionality_reduction",
                "representation": record.get("input_key"),
                "method": record.get("method"),
                "rows": record.get("rows"),
                "dimensions": record.get("output_dimensions"),
                "storage": "dense_float32_npy",
                "elapsed_seconds": record.get("elapsed_seconds"),
                "memory_mb": record.get("approx_output_memory_mb"),
                "finite_values": record.get("finite_values"),
                "fit_scope": record.get("fit_scope"),
                "semantic_quality_status": "not_evaluated_in_phase4",
            }
        )

    return pd.DataFrame(rows)


def write_stage3_metadata(
    path: Path,
    *,
    transformer_outputs: Mapping[str, str],
    reduction_outputs: Mapping[str, str],
    final_benchmark_path: Path,
) -> None:
    payload = {
        "stage": "04_03_transformer_embeddings_and_reduction",
        "status": "smoke_test_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "transformer_outputs": dict(transformer_outputs),
        "reduction_outputs": dict(reduction_outputs),
        "final_benchmark": str(final_benchmark_path),
        "leakage_boundary": {
            "frozen_embeddings": (
                "May be generated for all rows because external model weights and "
                "tokenizer are fixed and no corpus parameter is learned."
            ),
            "pca": "Fit on training embeddings only after the temporal split.",
            "truncated_svd": "Fit on training TF-IDF only after the temporal split.",
            "tfidf": "Fit vectorizers on training text only after the temporal split.",
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
