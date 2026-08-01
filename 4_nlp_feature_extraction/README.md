# Phase 4 — NLP Feature Extraction

Phase 4 turns the conservative text representations produced by Phase 3 into
model-ready features. This first stage is deliberately limited to validating
the handoff, comparing representations, and producing explainable row-level
numeric features.

## Stage 1 scope

- Input (read-only):
  `3_text_preprocessing/data/processed/cryptovision_v1_preprocessed.parquet`
- Audited representations: `text_title_description` and `Filtered_Text`
- Feature representation: `text_title_description` (an explicit, configurable
  decision recorded in `configs/row_features.json`)
- Outputs (generated and Gitignored):
  - `data/row_level_text_features.parquet`
  - `data/representation_audit.json`

The output contains `source_row_id` plus numeric features only. Features cover
length, words, digits, percentages, currency symbols, uppercase use,
punctuation, and a small documented group of finance-domain keyword flags.
Keyword indicators are presence flags—not sentiment scores or labels.

## Leakage boundary

Only `source_row_id` and the selected text representation enter feature
construction. Sentiment annotations, OHLCV fields, market moves, and outcome or
target columns are not inputs. The reusable function rejects text columns
outside the two approved Phase 3 representations.

## Run locally

1. Run Phase 3, or place the real file at
   `3_text_preprocessing/data/processed/cryptovision_v1_preprocessed.parquet`.
2. From the repository root, create an environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r 4_nlp_feature_extraction/requirements.txt
   ```

3. Start Jupyter from the repository root and run all cells in order:

   ```bash
   jupyter notebook 4_nlp_feature_extraction/notebooks/04_01_representation_audit_and_row_features.ipynb
   ```

The notebook stops with an actionable `FileNotFoundError` if the Gitignored
input is absent. It does not invent audit results.

## Tests

Tests use small deterministic in-memory fixtures and do not require the real
dataset:

```bash
pytest -q 4_nlp_feature_extraction/tests
```

## Later Phase 4 stages (not implemented here)

Future notebooks may fit **TF-IDF** baselines and contextual **BERT/FinBERT**
representations, with train-only fitting and leakage-safe evaluation. Sparse or
dense dimensionality reduction using **SVD/PCA** belongs after those
representations are built and only when justified by an experiment. None of
TF-IDF, BERT, FinBERT, PCA, or SVD is implemented in Stage 1.
