"""Frozen BERT/FinBERT feature extraction for Phase 4 Stage 3.

The module supports two execution modes:

1. small, reproducible smoke tests used by the educational notebook;
2. full local extraction with batching, float32 NumPy memmaps, and resume.

The transformer weights remain frozen. PCA and TruncatedSVD are implemented in
``reduction_features.py`` and are smoke-test recipes only until the temporal
train/validation/test split is defined.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .paths import find_project_root, resolve_project_path


VALID_TASKS = {"embedding", "sequence_classification"}
VALID_POOLING = {"masked_mean", "cls"}
VALID_DTYPES = {"float32"}
VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}
CANONICAL_SENTIMENT_LABELS = ("positive", "neutral", "negative")


def load_transformer_recipes(path: Path) -> dict[str, Any]:
    """Load and validate ``transformer_recipes.yaml``."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("transformer_recipes.yaml must contain a mapping at its root.")
    validate_transformer_recipes(config)
    return config


def validate_transformer_recipes(config: Mapping[str, Any]) -> None:
    """Validate fields required by transformer, reduction, and local full runs."""

    for section in ("input", "smoke_test", "models", "reduction", "full_run", "output"):
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    input_cfg = config["input"]
    if not isinstance(input_cfg.get("relative_path"), str) or not input_cfg["relative_path"].strip():
        raise ValueError("input.relative_path must be a non-empty string.")
    if not isinstance(input_cfg.get("id_column"), str) or not input_cfg["id_column"].strip():
        raise ValueError("input.id_column must be a non-empty string.")
    representations = input_cfg.get("representations")
    if not isinstance(representations, list) or not representations:
        raise ValueError("input.representations must contain at least one column name.")
    if not all(isinstance(item, str) and item.strip() for item in representations):
        raise ValueError("Every input representation must be a non-empty string.")
    if len(representations) != len(set(representations)):
        raise ValueError("input.representations must not contain duplicates.")

    smoke_cfg = config["smoke_test"]
    for key in ("sample_size", "batch_size"):
        if not isinstance(smoke_cfg.get(key), int) or smoke_cfg[key] <= 0:
            raise ValueError(f"smoke_test.{key} must be a positive integer.")
    if not isinstance(smoke_cfg.get("random_state"), int):
        raise ValueError("smoke_test.random_state must be an integer.")
    max_empty = smoke_cfg.get("max_empty_rows_per_representation")
    if not isinstance(max_empty, int) or max_empty < 0:
        raise ValueError(
            "smoke_test.max_empty_rows_per_representation must be a non-negative integer."
        )
    if smoke_cfg.get("device") not in VALID_DEVICES:
        raise ValueError("smoke_test.device must be auto, cpu, cuda, or mps.")
    if not isinstance(smoke_cfg.get("track_truncation"), bool):
        raise ValueError("smoke_test.track_truncation must be boolean.")

    models = config["models"]
    if not isinstance(models, Mapping) or not models:
        raise ValueError("models must contain at least one transformer recipe.")
    for name, spec in models.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(spec, Mapping):
            raise ValueError("Every model recipe needs a name and mapping value.")
        for key in ("model_id", "revision", "tokenizer_id", "tokenizer_revision"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"models.{name}.{key} must be a non-empty string.")
        if spec.get("task") not in VALID_TASKS:
            raise ValueError(f"models.{name}.task is unsupported.")
        if spec.get("pooling") not in VALID_POOLING:
            raise ValueError(f"models.{name}.pooling is unsupported.")
        max_length = spec.get("max_length")
        if not isinstance(max_length, int) or not 2 <= max_length <= 512:
            raise ValueError(f"models.{name}.max_length must be between 2 and 512.")
        if not isinstance(spec.get("truncation"), bool):
            raise ValueError(f"models.{name}.truncation must be boolean.")
        if spec.get("padding") not in {"longest", "max_length"}:
            raise ValueError(f"models.{name}.padding must be longest or max_length.")
        if spec.get("model_dtype") not in VALID_DTYPES:
            raise ValueError(f"models.{name}.model_dtype must be float32.")
        if spec.get("output_dtype") not in VALID_DTYPES:
            raise ValueError(f"models.{name}.output_dtype must be float32.")
        wants_sentiment = spec.get("sentiment_probabilities")
        if not isinstance(wants_sentiment, bool):
            raise ValueError(f"models.{name}.sentiment_probabilities must be boolean.")
        if wants_sentiment:
            if spec.get("task") != "sequence_classification":
                raise ValueError(
                    f"models.{name} requests sentiment probabilities but is not a "
                    "sequence-classification model."
                )
            order = spec.get("sentiment_output_order")
            if list(order or []) != list(CANONICAL_SENTIMENT_LABELS):
                raise ValueError(
                    f"models.{name}.sentiment_output_order must be "
                    "[positive, neutral, negative]."
                )

    reduction_cfg = config["reduction"]
    for name in ("pca", "truncated_svd"):
        if name not in reduction_cfg or not isinstance(reduction_cfg[name], Mapping):
            raise ValueError(f"reduction.{name} must be a mapping.")
        n_components = reduction_cfg[name].get("n_components")
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError(f"reduction.{name}.n_components must be positive.")
        if reduction_cfg[name].get("output_dtype") not in VALID_DTYPES:
            raise ValueError(f"reduction.{name}.output_dtype must be float32.")
    if not isinstance(reduction_cfg["pca"].get("random_state"), int):
        raise ValueError("reduction.pca.random_state must be an integer.")
    if not isinstance(reduction_cfg["truncated_svd"].get("random_state"), int):
        raise ValueError("reduction.truncated_svd.random_state must be an integer.")
    tfidf_recipes = reduction_cfg["truncated_svd"].get("tfidf_recipes")
    if not isinstance(tfidf_recipes, list) or not tfidf_recipes:
        raise ValueError("reduction.truncated_svd.tfidf_recipes must be non-empty.")

    full_cfg = config["full_run"]
    if not isinstance(full_cfg.get("batch_size"), int) or full_cfg["batch_size"] <= 0:
        raise ValueError("full_run.batch_size must be positive.")
    if full_cfg.get("device") not in VALID_DEVICES:
        raise ValueError("full_run.device must be auto, cpu, cuda, or mps.")
    for key in ("resume", "track_truncation"):
        if not isinstance(full_cfg.get(key), bool):
            raise ValueError(f"full_run.{key} must be boolean.")
    if not isinstance(full_cfg.get("output_directory_relative_path"), str):
        raise ValueError("full_run.output_directory_relative_path must be a string.")

    output_cfg = config["output"]
    required_outputs = (
        "transformer_smoke_directory_relative_path",
        "reduction_smoke_directory_relative_path",
        "transformer_benchmark_relative_path",
        "transformer_metadata_relative_path",
        "reduction_benchmark_relative_path",
        "metadata_relative_path",
        "final_benchmark_relative_path",
    )
    for key in required_outputs:
        if not isinstance(output_cfg.get(key), str) or not output_cfg[key].strip():
            raise ValueError(f"output.{key} must be a non-empty string.")


def normalize_transformer_texts(series: pd.Series) -> pd.Series:
    """Convert nulls to empty strings without changing row count or order."""

    return series.astype("string").fillna("").str.strip()


def make_transformer_smoke_sample(
    df: pd.DataFrame,
    *,
    id_column: str,
    representations: Sequence[str],
    sample_size: int,
    random_state: int,
    max_empty_rows_per_representation: int = 4,
) -> pd.DataFrame:
    """Build a fixed sample containing a few empty rows plus random normal rows.

    Stage 1 found 76 empty ``text_title_description`` rows and one empty/null
    ``Filtered_Text`` row. Including every empty row would make a small transformer
    sample unrepresentative, so this sampler keeps up to a configured number per
    representation and fills the remainder with a deterministic random sample.
    Selected positions are sorted before return, preserving original row order.
    """

    required = [id_column, *representations]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for the transformer sample: {missing}")
    if len(df) == 0:
        raise ValueError("Cannot sample an empty dataframe.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if max_empty_rows_per_representation < 0:
        raise ValueError("max_empty_rows_per_representation must be non-negative.")

    target_size = min(sample_size, len(df))
    rng = np.random.default_rng(random_state)
    selected_edge_positions: list[int] = []

    for representation in representations:
        empty_positions = np.flatnonzero(
            normalize_transformer_texts(df[representation]).eq("").to_numpy()
        )
        if len(empty_positions) > max_empty_rows_per_representation:
            empty_positions = np.sort(
                rng.choice(
                    empty_positions,
                    size=max_empty_rows_per_representation,
                    replace=False,
                )
            )
        selected_edge_positions.extend(int(value) for value in empty_positions)

    edge_positions = np.array(sorted(set(selected_edge_positions)), dtype=np.int64)
    if len(edge_positions) >= target_size:
        selected = edge_positions[:target_size]
    else:
        all_positions = np.arange(len(df), dtype=np.int64)
        remaining = np.setdiff1d(all_positions, edge_positions, assume_unique=True)
        needed = target_size - len(edge_positions)
        random_positions = rng.choice(remaining, size=needed, replace=False)
        selected = np.concatenate([edge_positions, random_positions])

    selected.sort()
    return df.iloc[selected].reset_index(drop=True).copy()


def select_torch_device(requested: str = "auto") -> str:
    """Resolve an available PyTorch device without silently selecting an unavailable one."""

    if requested not in VALID_DEVICES:
        raise ValueError(f"Unsupported device request: {requested}")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return requested


def _resolve_hub_revision(model_id: str, revision: str) -> str:
    """Resolve a branch/tag to a commit SHA when Hub access is available."""

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision)
        return str(info.sha or revision)
    except Exception:
        return revision


def load_transformer_bundle(
    model_name: str,
    spec: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Load a tokenizer and frozen model from Hugging Face.

    The requested branch/tag and the resolved commit SHA are both recorded. Model
    classes are selected by the configured task. No fine-tuning is performed.
    """

    try:
        import transformers
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Stage 3 requires transformers. Install Phase 4 requirements before "
            "running the transformer notebook or full extraction command."
        ) from exc

    model_id = str(spec["model_id"])
    revision = str(spec["revision"])
    tokenizer_id = str(spec["tokenizer_id"])
    tokenizer_revision = str(spec["tokenizer_revision"])
    resolved_model_revision = _resolve_hub_revision(model_id, revision)
    resolved_tokenizer_revision = _resolve_hub_revision(tokenizer_id, tokenizer_revision)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=resolved_tokenizer_revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if spec["task"] == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=resolved_model_revision,
            trust_remote_code=False,
        )
    else:
        model = AutoModel.from_pretrained(
            model_id,
            revision=resolved_model_revision,
            trust_remote_code=False,
        )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)

    hidden_size = int(getattr(model.config, "hidden_size"))
    runtime_commit = getattr(model.config, "_commit_hash", None)
    if runtime_commit:
        resolved_model_revision = str(runtime_commit)
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if tokenizer_commit:
        resolved_tokenizer_revision = str(tokenizer_commit)

    return {
        "model_name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "hidden_size": hidden_size,
        "requested_model_revision": revision,
        "resolved_model_revision": resolved_model_revision,
        "requested_tokenizer_revision": tokenizer_revision,
        "resolved_tokenizer_revision": resolved_tokenizer_revision,
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "transformers_version": transformers.__version__,
    }


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average non-padding token states for one embedding per input row."""

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp(min=1.0)
    return summed / denominator


def pool_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """Apply the configured CLS or masked-mean pooling strategy."""

    if pooling == "masked_mean":
        return masked_mean_pool(last_hidden_state, attention_mask)
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    raise ValueError(f"Unsupported pooling strategy: {pooling}")


def _token_lengths(tokenizer: Any, texts: Sequence[str]) -> np.ndarray:
    """Return untruncated token lengths using a fast-tokenizer-compatible API."""

    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_length=True,
    )
    lengths = encoded.get("length") if isinstance(encoded, Mapping) else None
    if lengths is None:
        input_ids = encoded["input_ids"]
        lengths = [len(row) for row in input_ids]
    return np.asarray(lengths, dtype=np.int32)


def _sentiment_indices(model: Any, output_order: Sequence[str]) -> list[int]:
    id2label = getattr(model.config, "id2label", {}) or {}
    normalized: dict[str, int] = {}
    for raw_index, raw_label in id2label.items():
        normalized[str(raw_label).strip().lower()] = int(raw_index)
    missing = [label for label in output_order if label not in normalized]
    if missing:
        raise ValueError(
            "The configured sequence-classification model is not compatible with "
            f"the requested sentiment labels. Missing: {missing}; id2label={id2label}"
        )
    return [normalized[label] for label in output_order]


def _infer_batch(
    texts: Sequence[str],
    *,
    bundle: Mapping[str, Any],
    spec: Mapping[str, Any],
    track_truncation: bool,
) -> dict[str, Any]:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    max_length = int(spec["max_length"])

    lengths = _token_lengths(tokenizer, texts) if track_truncation else None
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=max_length,
        truncation=bool(spec["truncation"]),
        padding=spec["padding"],
        return_attention_mask=True,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        if spec["task"] == "sequence_classification":
            outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
            hidden_state = outputs.hidden_states[-1]
        else:
            outputs = model(**model_inputs, return_dict=True)
            hidden_state = outputs.last_hidden_state
        pooled = pool_hidden_state(
            hidden_state,
            model_inputs["attention_mask"],
            str(spec["pooling"]),
        )

    embeddings = pooled.detach().to("cpu", dtype=torch.float32).numpy().astype(
        np.float32, copy=False
    )
    result: dict[str, Any] = {
        "embeddings": embeddings,
        "truncated_rows": int(np.count_nonzero(lengths > max_length)) if lengths is not None else None,
        "token_lengths": lengths,
    }

    if bool(spec.get("sentiment_probabilities")):
        indices = _sentiment_indices(model, spec["sentiment_output_order"])
        probabilities = torch.softmax(outputs.logits, dim=-1)[:, indices]
        result["sentiment_probabilities"] = (
            probabilities.detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32, copy=False)
        )
    else:
        result["sentiment_probabilities"] = None
    return result


def extract_transformer_arrays(
    texts: pd.Series,
    *,
    bundle: Mapping[str, Any],
    spec: Mapping[str, Any],
    batch_size: int,
    track_truncation: bool = True,
) -> dict[str, Any]:
    """Extract one dense float32 matrix and optional canonical sentiment probabilities."""

    normalized = normalize_transformer_texts(texts)
    batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    truncated_rows = 0
    token_length_values: list[np.ndarray] = []
    started = perf_counter()

    for start in range(0, len(normalized), batch_size):
        batch_texts = normalized.iloc[start : start + batch_size].tolist()
        batch = _infer_batch(
            batch_texts,
            bundle=bundle,
            spec=spec,
            track_truncation=track_truncation,
        )
        batches.append(batch["embeddings"])
        if batch["sentiment_probabilities"] is not None:
            probability_batches.append(batch["sentiment_probabilities"])
        if batch["truncated_rows"] is not None:
            truncated_rows += int(batch["truncated_rows"])
        if batch["token_lengths"] is not None:
            token_length_values.append(batch["token_lengths"])

    elapsed = perf_counter() - started
    embeddings = (
        np.vstack(batches).astype(np.float32, copy=False)
        if batches
        else np.empty((0, int(bundle["hidden_size"])), dtype=np.float32)
    )
    probabilities = (
        np.vstack(probability_batches).astype(np.float32, copy=False)
        if probability_batches
        else None
    )
    token_lengths = np.concatenate(token_length_values) if token_length_values else None

    if embeddings.shape != (len(normalized), int(bundle["hidden_size"])):
        raise RuntimeError("Transformer extraction changed row count or embedding dimension.")
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise RuntimeError("Transformer embeddings must be finite float32 values.")
    if probabilities is not None:
        if probabilities.shape != (len(normalized), 3):
            raise RuntimeError("Sentiment probabilities must have shape (rows, 3).")
        if probabilities.dtype != np.float32 or not np.isfinite(probabilities).all():
            raise RuntimeError("Sentiment probabilities must be finite float32 values.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Sentiment probabilities must sum to one per row.")

    norms = np.linalg.norm(embeddings, axis=1) if len(embeddings) else np.array([], dtype=np.float32)
    report = {
        "rows": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "shape": f"({embeddings.shape[0]}, {embeddings.shape[1]})",
        "dtype": str(embeddings.dtype),
        "elapsed_seconds": float(elapsed),
        "rows_per_second": float(len(embeddings) / elapsed) if elapsed > 0 else 0.0,
        "approx_embedding_memory_mb": float(embeddings.nbytes / (1024**2)),
        "finite_values": bool(np.isfinite(embeddings).all()),
        "zero_embedding_rows": int(np.count_nonzero(norms == 0.0)),
        "mean_l2_norm": float(norms.mean()) if len(norms) else 0.0,
        "std_l2_norm": float(norms.std()) if len(norms) else 0.0,
        "truncated_rows": int(truncated_rows) if track_truncation else None,
        "truncated_row_pct": (
            float(truncated_rows / len(normalized) * 100.0)
            if track_truncation and len(normalized)
            else None
        ),
        "median_untruncated_tokens": (
            float(np.median(token_lengths)) if token_lengths is not None and len(token_lengths) else None
        ),
        "max_untruncated_tokens": (
            int(token_lengths.max()) if token_lengths is not None and len(token_lengths) else None
        ),
        "sentiment_probabilities": probabilities is not None,
        "sentiment_probability_max_sum_error": (
            float(np.abs(probabilities.sum(axis=1) - 1.0).max())
            if probabilities is not None and len(probabilities)
            else None
        ),
    }
    return {
        "embeddings": embeddings,
        "sentiment_probabilities": probabilities,
        "report": report,
    }


def _bundle_metadata(bundle: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": spec["model_id"],
        "requested_model_revision": bundle.get("requested_model_revision", spec["revision"]),
        "resolved_model_revision": bundle.get("resolved_model_revision", spec["revision"]),
        "model_class": bundle.get("model_class", bundle["model"].__class__.__name__),
        "tokenizer_id": spec["tokenizer_id"],
        "requested_tokenizer_revision": bundle.get(
            "requested_tokenizer_revision", spec["tokenizer_revision"]
        ),
        "resolved_tokenizer_revision": bundle.get(
            "resolved_tokenizer_revision", spec["tokenizer_revision"]
        ),
        "tokenizer_class": bundle.get(
            "tokenizer_class", bundle["tokenizer"].__class__.__name__
        ),
        "hidden_size": int(bundle["hidden_size"]),
        "task": spec["task"],
        "max_length": int(spec["max_length"]),
        "pooling": spec["pooling"],
        "truncation": bool(spec["truncation"]),
        "padding": spec["padding"],
        "model_dtype": spec["model_dtype"],
        "output_dtype": spec["output_dtype"],
        "sentiment_probabilities": bool(spec["sentiment_probabilities"]),
        "sentiment_output_order": spec.get("sentiment_output_order"),
        "device": bundle["device"],
        "torch_version": torch.__version__,
        "transformers_version": bundle.get("transformers_version"),
        "python_version": platform.python_version(),
    }


def run_transformer_smoke_benchmark(
    sample_df: pd.DataFrame,
    *,
    id_column: str,
    representations: Sequence[str],
    models: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    device: str = "auto",
    track_truncation: bool = True,
    bundle_loader: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run BERT and FinBERT smoke extraction for every requested representation."""

    required = [id_column, *representations]
    missing = [column for column in required if column not in sample_df.columns]
    if missing:
        raise ValueError(f"Missing transformer benchmark columns: {missing}")

    resolved_device = select_torch_device(device)
    expected_ids = sample_df[id_column].reset_index(drop=True).copy()
    embeddings: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    model_metadata: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for model_name, spec in models.items():
        if bundle_loader is None:
            bundle = load_transformer_bundle(model_name, spec, device=resolved_device)
        else:
            bundle = dict(bundle_loader(model_name, spec))
            bundle.setdefault("device", resolved_device)
            bundle.setdefault("model_name", model_name)
            bundle.setdefault("hidden_size", int(bundle["model"].config.hidden_size))
        model_metadata[model_name] = _bundle_metadata(bundle, spec)

        for representation in representations:
            normalized = normalize_transformer_texts(sample_df[representation])
            result = extract_transformer_arrays(
                normalized,
                bundle=bundle,
                spec=spec,
                batch_size=batch_size,
                track_truncation=track_truncation,
            )
            key = f"{representation}__{model_name}"
            embeddings[key] = result["embeddings"]
            if result["sentiment_probabilities"] is not None:
                probabilities[key] = result["sentiment_probabilities"]
            rows.append(
                {
                    "representation": representation,
                    "model": model_name,
                    "model_id": spec["model_id"],
                    "resolved_revision": model_metadata[model_name]["resolved_model_revision"],
                    "tokenizer_id": spec["tokenizer_id"],
                    "max_length": int(spec["max_length"]),
                    "pooling": spec["pooling"],
                    "truncation": bool(spec["truncation"]),
                    "empty_input_rows": int(normalized.eq("").sum()),
                    "source_row_id_order_preserved": bool(
                        sample_df[id_column].reset_index(drop=True).equals(expected_ids)
                    ),
                    **result["report"],
                }
            )

        del bundle
        if resolved_device == "cuda":
            torch.cuda.empty_cache()

    return {
        "source_row_ids": expected_ids,
        "embeddings": embeddings,
        "sentiment_probabilities": probabilities,
        "benchmark": pd.DataFrame(rows),
        "model_metadata": model_metadata,
    }


def save_transformer_smoke_outputs(
    result: Mapping[str, Any],
    *,
    output_directory: Path,
    benchmark_path: Path,
    metadata_path: Path,
    id_column: str,
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Save small float32 arrays, one shared ID mapping, and exact model metadata."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    benchmark_path = Path(benchmark_path)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    ids_path = output_directory / "transformer_smoke_source_row_ids.csv"
    pd.DataFrame({id_column: result["source_row_ids"]}).to_csv(ids_path, index=False)
    output_files: dict[str, str] = {"source_row_ids": str(ids_path)}

    for key, matrix in result["embeddings"].items():
        path = output_directory / f"{key}_embeddings.npy"
        np.save(path, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
        output_files[f"embeddings::{key}"] = str(path)
    for key, matrix in result["sentiment_probabilities"].items():
        path = output_directory / f"{key}_sentiment_probabilities.npy"
        np.save(path, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
        output_files[f"sentiment::{key}"] = str(path)

    model_metadata_path = output_directory / "model_metadata.json"
    model_metadata_path.write_text(
        json.dumps(result["model_metadata"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_files["model_metadata"] = str(model_metadata_path)
    result["benchmark"].to_csv(benchmark_path, index=False)

    metadata = {
        "stage": "04_03_transformer_embeddings_and_reduction",
        "status": "transformer_smoke_test_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rows": int(len(result["source_row_ids"])),
        "representations": list(config["input"]["representations"]),
        "models": list(config["models"]),
        "leakage_note": (
            "Frozen external embeddings and FinBERT probabilities do not learn from "
            "this corpus. PCA and TruncatedSVD outputs created later in the notebook "
            "are smoke-test artifacts only and must be refit on training data."
        ),
        "outputs": output_files,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output_files["benchmark"] = str(benchmark_path)
    output_files["metadata"] = str(metadata_path)
    return output_files


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _job_fingerprint(
    *,
    ids: pd.Series,
    representation: str,
    model_name: str,
    spec: Mapping[str, Any],
) -> str:
    payload = {
        "rows": int(len(ids)),
        "first_id": None if len(ids) == 0 else str(ids.iloc[0]),
        "last_id": None if len(ids) == 0 else str(ids.iloc[-1]),
        "representation": representation,
        "model_name": model_name,
        "spec": deepcopy(dict(spec)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def extract_transformer_to_memmap(
    df: pd.DataFrame,
    *,
    id_column: str,
    representation: str,
    model_name: str,
    spec: Mapping[str, Any],
    output_directory: Path,
    batch_size: int,
    device: str = "auto",
    resume: bool = True,
    track_truncation: bool = False,
    bundle: Mapping[str, Any] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Extract full frozen embeddings with batch checkpoints and resume.

    ``max_batches`` exists for tests and controlled dry runs. Normal local runs
    leave it as ``None``.
    """

    for column in (id_column, representation):
        if column not in df.columns:
            raise ValueError(f"Missing full-run column: {column}")
    if len(df) == 0:
        raise ValueError("Cannot extract embeddings from an empty dataframe.")

    resolved_device = select_torch_device(device)
    own_bundle = bundle is None
    if bundle is None:
        bundle = load_transformer_bundle(model_name, spec, device=resolved_device)
    else:
        bundle = dict(bundle)
        bundle.setdefault("device", resolved_device)
        bundle.setdefault("hidden_size", int(bundle["model"].config.hidden_size))

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    key = f"{_safe_name(representation)}__{_safe_name(model_name)}"
    embedding_path = output_directory / f"{key}_embeddings.npy"
    probability_path = output_directory / f"{key}_sentiment_probabilities.npy"
    ids_path = output_directory / f"{key}_source_row_ids.csv"
    progress_path = output_directory / f"{key}_progress.json"
    metadata_path = output_directory / f"{key}_metadata.json"

    ids = df[id_column].reset_index(drop=True).copy()
    fingerprint = _job_fingerprint(
        ids=ids,
        representation=representation,
        model_name=model_name,
        spec=spec,
    )
    start_index = 0
    progress: dict[str, Any] = {}

    if resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing progress belongs to a different dataset or recipe.")
        if not embedding_path.is_file() or not ids_path.is_file():
            raise RuntimeError("Resume metadata exists but required output files are missing.")
        saved_ids = pd.read_csv(ids_path)[id_column]
        if saved_ids.astype(str).tolist() != ids.astype(str).tolist():
            raise RuntimeError("Saved source_row_id mapping does not match the current input.")
        start_index = int(progress.get("next_index", 0))
        if progress.get("status") == "completed" and start_index == len(df):
            return json.loads(metadata_path.read_text(encoding="utf-8"))

    hidden_size = int(bundle["hidden_size"])
    if start_index == 0:
        pd.DataFrame({id_column: ids}).to_csv(ids_path, index=False)
        embeddings_memmap = np.lib.format.open_memmap(
            embedding_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(df), hidden_size),
        )
        probabilities_memmap = None
        if bool(spec.get("sentiment_probabilities")):
            probabilities_memmap = np.lib.format.open_memmap(
                probability_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(df), 3),
            )
    else:
        embeddings_memmap = np.lib.format.open_memmap(embedding_path, mode="r+")
        probabilities_memmap = (
            np.lib.format.open_memmap(probability_path, mode="r+")
            if bool(spec.get("sentiment_probabilities"))
            else None
        )

    normalized = normalize_transformer_texts(df[representation])
    total_truncated = int(progress.get("truncated_rows", 0))
    total_elapsed = float(progress.get("elapsed_seconds", 0.0))
    processed_batches = 0

    for batch_start in range(start_index, len(df), batch_size):
        if max_batches is not None and processed_batches >= max_batches:
            break
        batch_end = min(batch_start + batch_size, len(df))
        started = perf_counter()
        batch = _infer_batch(
            normalized.iloc[batch_start:batch_end].tolist(),
            bundle=bundle,
            spec=spec,
            track_truncation=track_truncation,
        )
        embeddings_memmap[batch_start:batch_end] = batch["embeddings"]
        embeddings_memmap.flush()
        if probabilities_memmap is not None:
            probabilities_memmap[batch_start:batch_end] = batch["sentiment_probabilities"]
            probabilities_memmap.flush()
        total_elapsed += perf_counter() - started
        if batch["truncated_rows"] is not None:
            total_truncated += int(batch["truncated_rows"])
        processed_batches += 1
        progress = {
            "status": "running",
            "fingerprint": fingerprint,
            "next_index": int(batch_end),
            "total_rows": int(len(df)),
            "truncated_rows": int(total_truncated),
            "elapsed_seconds": float(total_elapsed),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json_write(progress_path, progress)

    next_index = int(progress.get("next_index", start_index))
    completed = next_index >= len(df)
    progress["status"] = "completed" if completed else "paused"
    _atomic_json_write(progress_path, progress)

    metadata = {
        "stage": "04_03_transformer_embeddings_and_reduction",
        "status": progress["status"],
        "representation": representation,
        "model": model_name,
        "rows": int(len(df)),
        "completed_rows": next_index,
        "dimensions": hidden_size,
        "dtype": "float32",
        "batch_size": int(batch_size),
        "resume_enabled": bool(resume),
        "track_truncation": bool(track_truncation),
        "truncated_rows": int(total_truncated) if track_truncation else None,
        "elapsed_seconds": float(total_elapsed),
        "model_metadata": _bundle_metadata(bundle, spec),
        "outputs": {
            "embeddings": str(embedding_path),
            "source_row_ids": str(ids_path),
            "progress": str(progress_path),
            "sentiment_probabilities": (
                str(probability_path) if probabilities_memmap is not None else None
            ),
        },
    }
    _atomic_json_write(metadata_path, metadata)
    if own_bundle and resolved_device == "cuda":
        torch.cuda.empty_cache()
    return metadata


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable frozen transformer extraction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()
    project_root = (args.project_root or find_project_root()).resolve()
    config = load_transformer_recipes(args.config)
    if args.model not in config["models"]:
        raise KeyError(f"Unknown model recipe: {args.model}")
    if args.representation not in config["input"]["representations"]:
        raise KeyError(f"Unknown representation: {args.representation}")

    input_path = resolve_project_path(project_root, config["input"]["relative_path"])
    output_directory = resolve_project_path(
        project_root, config["full_run"]["output_directory_relative_path"]
    )
    columns = [config["input"]["id_column"], args.representation]
    frame = pd.read_parquet(input_path, columns=columns)
    metadata = extract_transformer_to_memmap(
        frame,
        id_column=config["input"]["id_column"],
        representation=args.representation,
        model_name=args.model,
        spec=config["models"][args.model],
        output_directory=output_directory,
        batch_size=args.batch_size or config["full_run"]["batch_size"],
        device=args.device or config["full_run"]["device"],
        resume=not args.no_resume and config["full_run"]["resume"],
        track_truncation=config["full_run"]["track_truncation"],
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
