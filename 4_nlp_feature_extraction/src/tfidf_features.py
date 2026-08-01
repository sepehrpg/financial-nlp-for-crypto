"""Reusable TF-IDF helpers for Phase 4 Stage 2.

The functions in this module are intentionally small and notebook-friendly.
They support reproducible smoke tests only. Final vectorizers must be fit on the
training split after the temporal split is defined.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


VALID_RECIPE_KINDS = {"single", "combined"}
VALID_ANALYZERS = {"word", "char", "char_wb"}


def load_feature_recipes(path: Path) -> dict[str, Any]:
    """Load and validate the Stage 2 YAML configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError("feature_recipes.yaml must contain a mapping at its root.")

    validate_feature_recipes(config)
    return config


def validate_feature_recipes(config: Mapping[str, Any]) -> None:
    """Validate only the fields needed by the Stage 2 smoke test."""

    for section in ("input", "smoke_test", "recipes", "output"):
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    input_cfg = config["input"]
    representations = input_cfg.get("representations", [])
    if not isinstance(input_cfg.get("id_column"), str) or not input_cfg["id_column"].strip():
        raise ValueError("input.id_column must be a non-empty string.")
    if not isinstance(representations, list) or not representations:
        raise ValueError("input.representations must contain at least one column name.")
    if len(representations) != len(set(representations)):
        raise ValueError("input.representations must not contain duplicates.")

    smoke_cfg = config["smoke_test"]
    if not isinstance(smoke_cfg.get("sample_size"), int) or smoke_cfg["sample_size"] <= 0:
        raise ValueError("smoke_test.sample_size must be a positive integer.")
    if not isinstance(smoke_cfg.get("random_state"), int):
        raise ValueError("smoke_test.random_state must be an integer.")
    if not isinstance(smoke_cfg.get("include_all_empty_rows"), bool):
        raise ValueError("smoke_test.include_all_empty_rows must be boolean.")

    recipes = config["recipes"]
    if not isinstance(recipes, dict) or not recipes:
        raise ValueError("recipes must contain at least one TF-IDF recipe.")

    for name, recipe in recipes.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(recipe, dict):
            raise ValueError("Every recipe needs a non-empty name and a mapping value.")
        kind = recipe.get("kind")
        if kind not in VALID_RECIPE_KINDS:
            raise ValueError(f"Recipe {name!r} has unsupported kind: {kind!r}")
        if kind == "single":
            params = recipe.get("vectorizer")
            if not isinstance(params, dict):
                raise ValueError(f"Single recipe {name!r} needs vectorizer settings.")
            if params.get("analyzer") not in VALID_ANALYZERS:
                raise ValueError(f"Recipe {name!r} has an unsupported analyzer.")
            ngram_range = params.get("ngram_range")
            if (
                not isinstance(ngram_range, list)
                or len(ngram_range) != 2
                or not all(isinstance(value, int) and value > 0 for value in ngram_range)
                or ngram_range[0] > ngram_range[1]
            ):
                raise ValueError(f"Recipe {name!r} needs a valid two-value ngram_range.")
        else:
            word_name = recipe.get("word_recipe")
            char_name = recipe.get("character_recipe")
            if word_name not in recipes or char_name not in recipes:
                raise ValueError(f"Combined recipe {name!r} references missing recipes.")
            if recipes[word_name].get("kind") != "single" or recipes[char_name].get("kind") != "single":
                raise ValueError(f"Combined recipe {name!r} must reference single recipes.")


def normalize_texts(series: pd.Series) -> pd.Series:
    """Return strings suitable for scikit-learn while preserving row count/order."""

    return series.astype("string").fillna("").str.strip()


def make_smoke_sample(
    df: pd.DataFrame,
    *,
    id_column: str,
    representations: Sequence[str],
    sample_size: int,
    random_state: int,
    include_all_empty_rows: bool = True,
) -> pd.DataFrame:
    """Create a fixed sample and keep the original row order.

    Empty rows are included first so zero-vector behavior is always audited.
    Remaining rows are selected with a fixed random seed. Selected positions are
    sorted before returning, so source-row order is preserved.
    """

    required = [id_column, *representations]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for the smoke sample: {missing}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if len(df) == 0:
        raise ValueError("Cannot sample an empty dataframe.")

    target_size = min(sample_size, len(df))
    all_positions = np.arange(len(df), dtype=np.int64)
    edge_positions = np.array([], dtype=np.int64)

    if include_all_empty_rows:
        empty_mask = pd.Series(False, index=df.index)
        for representation in representations:
            empty_mask |= normalize_texts(df[representation]).eq("")
        edge_positions = np.flatnonzero(empty_mask.to_numpy())

    if len(edge_positions) >= target_size:
        selected = edge_positions[:target_size]
    else:
        remaining = np.setdiff1d(all_positions, edge_positions, assume_unique=True)
        remaining_needed = target_size - len(edge_positions)
        rng = np.random.default_rng(random_state)
        random_positions = rng.choice(remaining, size=remaining_needed, replace=False)
        selected = np.concatenate([edge_positions, random_positions])

    selected.sort()
    return df.iloc[selected].reset_index(drop=True).copy()


def _vectorizer_parameters(settings: Mapping[str, Any]) -> dict[str, Any]:
    params = deepcopy(dict(settings))
    params["ngram_range"] = tuple(params["ngram_range"])
    dtype = params.get("dtype", "float32")
    if dtype == "float32":
        params["dtype"] = np.float32
    elif dtype == "float64":
        params["dtype"] = np.float64
    else:
        raise ValueError(f"Unsupported TF-IDF dtype: {dtype!r}")
    return params


def build_vectorizer(settings: Mapping[str, Any]) -> TfidfVectorizer:
    """Build one scikit-learn vectorizer from YAML-compatible settings."""

    return TfidfVectorizer(**_vectorizer_parameters(settings))


def fit_transform_recipe(
    texts: pd.Series,
    *,
    recipe_name: str,
    recipes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit one smoke-test recipe and return a CSR matrix plus metadata."""

    if recipe_name not in recipes:
        raise KeyError(f"Unknown TF-IDF recipe: {recipe_name}")

    recipe = recipes[recipe_name]
    kind = recipe["kind"]

    if kind == "single":
        vectorizer = build_vectorizer(recipe["vectorizer"])
        matrix = vectorizer.fit_transform(texts.tolist()).tocsr()
        names = vectorizer.get_feature_names_out().astype(str).tolist()
        return {
            "matrix": matrix,
            "feature_names": names,
            "vectorizers": {recipe_name: vectorizer},
        }

    word_name = recipe["word_recipe"]
    char_name = recipe["character_recipe"]
    word_result = fit_transform_recipe(texts, recipe_name=word_name, recipes=recipes)
    char_result = fit_transform_recipe(texts, recipe_name=char_name, recipes=recipes)
    matrix = sparse.hstack(
        [word_result["matrix"], char_result["matrix"]],
        format="csr",
        dtype=np.float32,
    )
    names = [
        *[f"word__{name}" for name in word_result["feature_names"]],
        *[f"char__{name}" for name in char_result["feature_names"]],
    ]
    vectorizers = {**word_result["vectorizers"], **char_result["vectorizers"]}
    return {"matrix": matrix, "feature_names": names, "vectorizers": vectorizers}


def sparse_matrix_memory_bytes(matrix: sparse.spmatrix) -> int:
    """Approximate in-memory bytes used by a CSR/CSC sparse matrix."""

    compressed = matrix.tocsr()
    return int(
        compressed.data.nbytes
        + compressed.indices.nbytes
        + compressed.indptr.nbytes
    )


def sparse_matrix_report(matrix: sparse.spmatrix, *, elapsed_seconds: float) -> dict[str, Any]:
    """Return the diagnostics requested for one TF-IDF experiment."""

    csr = matrix.tocsr()
    rows, columns = csr.shape
    total_cells = rows * columns
    zero_vectors = int(np.count_nonzero(np.diff(csr.indptr) == 0))
    finite = bool(np.isfinite(csr.data).all())
    sparsity_pct = 100.0 if total_cells == 0 else (1.0 - csr.nnz / total_cells) * 100.0

    return {
        "rows": int(rows),
        "features": int(columns),
        "shape": f"({rows}, {columns})",
        "nonzero_values": int(csr.nnz),
        "sparsity_pct": float(sparsity_pct),
        "zero_vectors": zero_vectors,
        "zero_vector_pct": float(zero_vectors / rows * 100.0) if rows else 0.0,
        "elapsed_seconds": float(elapsed_seconds),
        "approx_sparse_memory_mb": sparse_matrix_memory_bytes(csr) / (1024**2),
        "is_sparse_csr": bool(sparse.isspmatrix_csr(csr)),
        "finite_values": finite,
    }


def run_tfidf_smoke_benchmark(
    sample_df: pd.DataFrame,
    *,
    id_column: str,
    representations: Sequence[str],
    recipes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run every recipe on every representation without creating dense matrices."""

    if id_column not in sample_df.columns:
        raise ValueError(f"Missing identifier column: {id_column}")

    matrices: dict[str, sparse.csr_matrix] = {}
    feature_names: dict[str, list[str]] = {}
    vectorizers: dict[str, dict[str, TfidfVectorizer]] = {}
    rows: list[dict[str, Any]] = []
    expected_ids = sample_df[id_column].reset_index(drop=True)

    for representation in representations:
        if representation not in sample_df.columns:
            raise ValueError(f"Missing representation: {representation}")
        texts = normalize_texts(sample_df[representation])
        empty_input_rows = int(texts.eq("").sum())

        for recipe_name in recipes:
            started = perf_counter()
            result = fit_transform_recipe(
                texts,
                recipe_name=recipe_name,
                recipes=recipes,
            )
            elapsed = perf_counter() - started
            matrix = result["matrix"].tocsr()
            if matrix.shape[0] != len(sample_df):
                raise RuntimeError("TF-IDF changed the number of rows.")
            if not np.isfinite(matrix.data).all():
                raise RuntimeError("TF-IDF produced NaN or infinite values.")

            key = f"{representation}__{recipe_name}"
            matrices[key] = matrix
            feature_names[key] = result["feature_names"]
            vectorizers[key] = result["vectorizers"]
            report = sparse_matrix_report(matrix, elapsed_seconds=elapsed)
            rows.append({
                "representation": representation,
                "recipe": recipe_name,
                "empty_input_rows": empty_input_rows,
                "source_row_id_order_preserved": bool(
                    sample_df[id_column].reset_index(drop=True).equals(expected_ids)
                ),
                **report,
            })

    return {
        "source_row_ids": expected_ids.copy(),
        "matrices": matrices,
        "feature_names": feature_names,
        "vectorizers": vectorizers,
        "benchmark": pd.DataFrame(rows),
    }


def save_tfidf_smoke_outputs(
    result: Mapping[str, Any],
    *,
    output_directory: Path,
    benchmark_path: Path,
    metadata_path: Path,
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Save smoke matrices in sparse NPZ format and their row/feature mappings."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    benchmark_path = Path(benchmark_path)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    ids_path = output_directory / "smoke_sample_source_row_ids.csv"
    pd.DataFrame({config["input"]["id_column"]: result["source_row_ids"]}).to_csv(
        ids_path, index=False
    )

    output_files: dict[str, str] = {"source_row_ids": str(ids_path)}
    for key, matrix in result["matrices"].items():
        matrix_path = output_directory / f"{key}.npz"
        names_path = output_directory / f"{key}_feature_names.json"
        vectorizers_path = output_directory / f"{key}_vectorizers.joblib"
        sparse.save_npz(matrix_path, matrix.tocsr(), compressed=True)
        names_path.write_text(
            json.dumps(result["feature_names"][key], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        joblib.dump(result["vectorizers"][key], vectorizers_path)
        output_files[f"matrix::{key}"] = str(matrix_path)
        output_files[f"features::{key}"] = str(names_path)
        output_files[f"vectorizers::{key}"] = str(vectorizers_path)

    result["benchmark"].to_csv(benchmark_path, index=False)
    metadata = {
        "stage": "04_02_tfidf_feature_extraction",
        "status": "smoke_test_only",
        "leakage_warning": (
            "These vectorizers were fit only for a fixed smoke test. Do not use them "
            "as final model features. After the temporal split, fit on train only and "
            "transform validation/test without refitting."
        ),
        "sample_rows": int(len(result["source_row_ids"])),
        "representations": list(config["input"]["representations"]),
        "recipes": list(config["recipes"]),
        "outputs": output_files,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output_files["benchmark"] = str(benchmark_path)
    output_files["metadata"] = str(metadata_path)
    return output_files
