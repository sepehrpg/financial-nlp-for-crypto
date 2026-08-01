from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.transformer_features import (
    extract_transformer_to_memmap,
    load_transformer_recipes,
    make_transformer_smoke_sample,
    masked_mean_pool,
    run_transformer_smoke_benchmark,
    save_transformer_smoke_outputs,
)


class FakeTokenizer:
    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_length=False,
        max_length=None,
        return_attention_mask=False,
        return_tensors=None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        sequences = []
        for text in texts:
            tokens = [((sum(map(ord, token)) % 47) + 1) for token in str(text).split()]
            sequence = ([101] if add_special_tokens else []) + tokens + (
                [102] if add_special_tokens else []
            )
            if truncation and max_length is not None:
                sequence = sequence[:max_length]
            sequences.append(sequence)
        lengths = [len(sequence) for sequence in sequences]
        if return_tensors is None:
            result = {"input_ids": sequences}
            if return_length:
                result["length"] = lengths
            return result

        if padding == "max_length":
            padded_length = int(max_length)
        else:
            padded_length = max(lengths) if lengths else 0
        ids = []
        masks = []
        for sequence in sequences:
            pad = padded_length - len(sequence)
            ids.append(sequence + [0] * pad)
            masks.append([1] * len(sequence) + [0] * pad)
        result = {"input_ids": torch.tensor(ids, dtype=torch.long)}
        if return_attention_mask:
            result["attention_mask"] = torch.tensor(masks, dtype=torch.long)
        return result


class FakeBert(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, _commit_hash="fake-bert-sha")

    def forward(self, input_ids, attention_mask, return_dict=True):
        offsets = torch.arange(
            self.config.hidden_size, device=input_ids.device, dtype=torch.float32
        )
        hidden = input_ids.to(torch.float32).unsqueeze(-1) / 100.0 + offsets
        return SimpleNamespace(last_hidden_state=hidden)


class FakeFinBert(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            _commit_hash="fake-finbert-sha",
            id2label={0: "positive", 1: "negative", 2: "neutral"},
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        output_hidden_states=True,
        return_dict=True,
    ):
        offsets = torch.arange(
            self.config.hidden_size, device=input_ids.device, dtype=torch.float32
        )
        hidden = input_ids.to(torch.float32).unsqueeze(-1) / 120.0 + offsets
        score = (input_ids * attention_mask).sum(dim=1).to(torch.float32) / 100.0
        logits = torch.stack([score, -score, torch.zeros_like(score)], dim=1)
        return SimpleNamespace(hidden_states=(hidden,), logits=logits)


def _bundle_loader(model_name, spec):
    model = FakeFinBert() if model_name == "finbert" else FakeBert()
    return {
        "model_name": model_name,
        "model": model,
        "tokenizer": FakeTokenizer(),
        "device": "cpu",
        "hidden_size": model.config.hidden_size,
        "requested_model_revision": spec["revision"],
        "resolved_model_revision": model.config._commit_hash,
        "requested_tokenizer_revision": spec["tokenizer_revision"],
        "resolved_tokenizer_revision": "fake-tokenizer-sha",
        "model_class": model.__class__.__name__,
        "tokenizer_class": "FakeTokenizer",
        "transformers_version": "test-double",
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_id": [10, 13, 20, 21, 35, 40, 51, 66, 72, 90],
            "text_title_description": [
                "Bitcoin ETF approval lifts prices",
                "",
                "Federal Reserve holds interest rates",
                "Crypto exchange reports a security incident",
                "BTC falls after regulatory lawsuit",
                "Institutional demand supports the rally",
                "Will bitcoin volatility increase?",
                "Options markets expect a sharp move",
                "Fintech company expands crypto services",
                "Bitcoin remains stable",
            ],
            "Filtered_Text": [
                "bitcoin etf approval lift price",
                "short retained publisher text",
                "federal reserve hold interest rate",
                "crypto exchange security incident",
                "btc fall regulatory lawsuit",
                "institutional demand support rally",
                "bitcoin volatility increase",
                "",
                "fintech company expand crypto service",
                "bitcoin remain stable",
            ],
        }
    )


def test_masked_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor(
        [[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]], dtype=torch.float32
    )
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    pooled = masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 4.0]]))


def test_transformer_sample_is_reproducible_and_balanced() -> None:
    frame = pd.concat([_frame()] * 4, ignore_index=True)
    frame["source_row_id"] = range(100, 100 + len(frame))
    first = make_transformer_smoke_sample(
        frame,
        id_column="source_row_id",
        representations=["text_title_description", "Filtered_Text"],
        sample_size=20,
        random_state=42,
        max_empty_rows_per_representation=2,
    )
    second = make_transformer_smoke_sample(
        frame,
        id_column="source_row_id",
        representations=["text_title_description", "Filtered_Text"],
        sample_size=20,
        random_state=42,
        max_empty_rows_per_representation=2,
    )
    assert first["source_row_id"].tolist() == second["source_row_id"].tolist()
    assert first["source_row_id"].is_monotonic_increasing
    assert first["text_title_description"].eq("").any()
    assert first["Filtered_Text"].eq("").any()
    assert len(first) == 20


def test_transformer_smoke_outputs_are_float32_finite_and_aligned(phase4_dir: Path) -> None:
    config = load_transformer_recipes(
        phase4_dir / "configs" / "transformer_recipes.yaml"
    )
    for spec in config["models"].values():
        spec["max_length"] = 8
    result = run_transformer_smoke_benchmark(
        _frame(),
        id_column="source_row_id",
        representations=config["input"]["representations"],
        models=config["models"],
        batch_size=3,
        device="cpu",
        track_truncation=True,
        bundle_loader=_bundle_loader,
    )

    assert result["source_row_ids"].tolist() == _frame()["source_row_id"].tolist()
    assert len(result["embeddings"]) == 4
    assert len(result["sentiment_probabilities"]) == 2
    for matrix in result["embeddings"].values():
        assert matrix.shape == (len(_frame()), 8)
        assert matrix.dtype == np.float32
        assert np.isfinite(matrix).all()
    for probabilities in result["sentiment_probabilities"].values():
        assert probabilities.shape == (len(_frame()), 3)
        assert probabilities.dtype == np.float32
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result["benchmark"]["source_row_id_order_preserved"].all()
    assert result["benchmark"]["finite_values"].all()


def test_transformer_smoke_files_round_trip(tmp_path: Path, phase4_dir: Path) -> None:
    config = load_transformer_recipes(
        phase4_dir / "configs" / "transformer_recipes.yaml"
    )
    result = run_transformer_smoke_benchmark(
        _frame(),
        id_column="source_row_id",
        representations=config["input"]["representations"],
        models=config["models"],
        batch_size=4,
        device="cpu",
        track_truncation=True,
        bundle_loader=_bundle_loader,
    )
    files = save_transformer_smoke_outputs(
        result,
        output_directory=tmp_path / "transformer",
        benchmark_path=tmp_path / "transformer_benchmark.csv",
        metadata_path=tmp_path / "metadata.json",
        id_column="source_row_id",
        config=config,
    )
    loaded = np.load(files["embeddings::text_title_description__bert"])
    assert loaded.dtype == np.float32
    assert loaded.shape == (len(_frame()), 8)
    ids = pd.read_csv(files["source_row_ids"])["source_row_id"].tolist()
    assert ids == _frame()["source_row_id"].tolist()


def test_full_memmap_job_can_pause_and_resume(tmp_path: Path, phase4_dir: Path) -> None:
    config = load_transformer_recipes(
        phase4_dir / "configs" / "transformer_recipes.yaml"
    )
    spec = config["models"]["finbert"]
    bundle = _bundle_loader("finbert", spec)
    paused = extract_transformer_to_memmap(
        _frame(),
        id_column="source_row_id",
        representation="text_title_description",
        model_name="finbert",
        spec=spec,
        output_directory=tmp_path,
        batch_size=3,
        device="cpu",
        resume=True,
        track_truncation=True,
        bundle=bundle,
        max_batches=1,
    )
    assert paused["status"] == "paused"
    assert paused["completed_rows"] == 3

    completed = extract_transformer_to_memmap(
        _frame(),
        id_column="source_row_id",
        representation="text_title_description",
        model_name="finbert",
        spec=spec,
        output_directory=tmp_path,
        batch_size=3,
        device="cpu",
        resume=True,
        track_truncation=True,
        bundle=bundle,
    )
    assert completed["status"] == "completed"
    assert completed["completed_rows"] == len(_frame())
    embeddings = np.load(completed["outputs"]["embeddings"], mmap_mode="r")
    probabilities = np.load(
        completed["outputs"]["sentiment_probabilities"], mmap_mode="r"
    )
    assert embeddings.shape == (len(_frame()), 8)
    assert probabilities.shape == (len(_frame()), 3)
    assert embeddings.dtype == np.float32
    assert probabilities.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
