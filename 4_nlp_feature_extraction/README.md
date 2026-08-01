# Phase 4 — NLP Feature Extraction

This folder contains the feature-extraction phase of the Financial NLP for
Crypto project. Stage 1 audits the text representations produced by Phase 3 and
creates deterministic row-level numeric features.

## Stage 1 scope

Implemented notebook:

```text
04_01_representation_audit_and_row_features.ipynb
```

Stage 1 performs the following work:

1. reads the Phase 3 Parquet dataset;
2. records the current schema and row count;
3. validates `source_row_id` and requested text representations;
4. compares `text_title_description` and `Filtered_Text` when available;
5. calculates empty-text and text-length statistics;
6. creates corpus-independent, explainable numeric features;
7. saves row-level features and validation reports.

TF-IDF, BERT, FinBERT, PCA, and TruncatedSVD are deliberately postponed to the
later Phase 4 notebooks.

## Validation philosophy

The dataset is allowed to evolve. Normal changes must be reported rather than
turned into artificial failures.

The current row count is therefore **not hard-coded**. Every run records the
observed row count in the validation report and metadata. An optional
`reference_row_count` may be configured for historical comparison, but a
mismatch is non-blocking.

By default, the pipeline continues when it detects:

- a changed row count;
- duplicate `source_row_id` values;
- missing identifier values;
- a non-integer identifier dtype;
- non-monotonic identifier order;
- complete duplicate rows;
- one missing requested text representation.

These conditions are written to CSV and JSON metadata for review. Phase 4 does
not silently repair or delete affected rows.

The pipeline stops only when continuing would make the requested output
impossible or structurally invalid, including:

- the input file is missing or unreadable;
- the dataset contains zero rows;
- `source_row_id` is absent;
- none of the requested text representations is available;
- feature extraction changes row count or row order;
- generated feature values contain non-numeric columns, NaN, or infinity;
- the Parquet output cannot be written.

Even for blocking input checks, the pipeline writes a validation report and
blocked-run metadata before raising an exception whenever possible.

## Phase 3 handoff

Default input:

```text
3_text_preprocessing/data/processed/cryptovision_v1_preprocessed.parquet
```

Required provenance column:

```text
source_row_id
```

Requested representation candidates:

```text
text_title_description
Filtered_Text
```

If one representation is missing, Stage 1 uses the other and records the skipped
column. If both are missing, no text feature can be constructed and the run is
blocked.

## Folder structure

```text
4_nlp_feature_extraction/
├── README.md
├── requirements.txt
├── configs/
│   └── row_feature_recipe.yaml
├── data/
│   ├── reports/
│   │   └── .gitkeep
│   └── row_features/
│       └── .gitkeep
├── notebooks/
│   └── 04_01_representation_audit_and_row_features.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── paths.py
│   ├── pipeline.py
│   ├── representation_audit.py
│   ├── row_features.py
│   └── validation.py
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_pipeline.py
    ├── test_representation_audit.py
    ├── test_row_features.py
    └── test_validation.py
```

## Generated artifacts

```text
4_nlp_feature_extraction/data/row_features/
└── row_level_text_features.parquet

4_nlp_feature_extraction/data/reports/
├── phase3_contract_validation.csv
├── row_feature_validation.csv
├── representation_comparison.csv
├── representation_comparison.json
└── stage1_run_metadata.json
```

`stage1_run_metadata.json` records the current row count, requested and used
representations, skipped representations, validation issue counts, output paths,
and elapsed time.

## Representation comparison

For every available representation, the audit reports:

- total rows;
- null, empty, and non-empty counts;
- empty percentage;
- word-count median, configured percentile, and maximum;
- character-count median, configured percentile, and maximum;
- very-short text counts and percentages;
- non-empty text medians.

`text_title_description` is project-controlled conservative text. `Filtered_Text`
is publisher-provided and may reflect preprocessing choices outside this
project. Stage 1 measures both rather than declaring one universally superior.

## Explainable row-level features

The feature recipe is fixed before reading the corpus. It does not learn a
vocabulary, document frequency, embedding space, or dimensionality reduction.

For each available representation, the pipeline creates:

- empty indicator;
- character count;
- word count;
- number count;
- percentage count;
- currency-symbol count;
- uppercase character count;
- alphabetic character count;
- uppercase ratio;
- question-mark count;
- exclamation-mark count;
- fixed financial keyword-group counts and presence flags;
- total fixed financial keyword count.

The default keyword groups cover crypto assets, market direction, regulation,
instruments, macroeconomics, and risk events. These are transparent text signals,
not labels or sentiment truth.

## Configuration

Edit:

```text
4_nlp_feature_extraction/configs/row_feature_recipe.yaml
```

For an evolving dataset, keep:

```yaml
input:
  reference_row_count: null
```

A historical reference can be added for monitoring by replacing `null` with
the row count recorded by a previous approved run. A future difference from
that value will appear in the report but will not stop the pipeline.

## Running the notebook

From the repository root:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r 4_nlp_feature_extraction/requirements.txt
jupyter notebook 4_nlp_feature_extraction/notebooks/04_01_representation_audit_and_row_features.ipynb
```

Run cells from top to bottom.

## Running the pipeline directly

From the repository root, enter the Phase 4 folder and run the reusable module:

```bash
cd 4_nlp_feature_extraction
python -m src.pipeline --config configs/row_feature_recipe.yaml
```

## Running tests

```bash
cd 4_nlp_feature_extraction
pytest -q
```

Tests use synthetic data and do not require the large Phase 3 Parquet file.

## Leakage boundary

Stage 1 features are row-local and corpus-independent. Publisher sentiment,
OHLCV columns, `Market_Move`, and future market outcomes are not added to the
text-only feature matrix.

Later methods such as TF-IDF, PCA, and TruncatedSVD learn parameters from
multiple rows. Their final fit must occur only after the time-aware split in
Phase 6.

## Stage 2 — TF-IDF smoke-test extraction

Stage 2 is implemented in:

```text
notebooks/04_02_tfidf_feature_extraction.ipynb
src/tfidf_features.py
configs/feature_recipes.yaml
```

It compares six smoke-test experiments:

| Representation | Word TF-IDF | Character TF-IDF | Word + Character |
|---|---:|---:|---:|
| `text_title_description` | yes | yes | yes |
| `Filtered_Text` | yes | yes | yes |

The smoke sample is deterministic. It includes rows that are empty in either
representation before filling the remaining sample positions with a fixed random
seed. This is important because Stage 1 found:

- 76 empty `text_title_description` rows;
- 1 null/empty `Filtered_Text` row.

Empty rows are retained and reported as possible zero-vectors. They are not
silently removed.

### Sparse artifacts

The notebook writes matrices with `scipy.sparse.save_npz`; it never converts the
TF-IDF matrix to a dense dataframe or dense NumPy array. Each matrix has:

- an `.npz` sparse matrix;
- a JSON feature-name mapping;
- a Joblib file containing the smoke-test vectorizer(s);
- one shared CSV preserving `source_row_id` row order.

### Leakage boundary

The saved vectorizers and matrices are diagnostic smoke-test artifacts only.
They must not be used as final model features. After Phase 6 creates the temporal
split:

1. fit the vectorizer on the training text only;
2. transform validation and test text without refitting;
3. fit any TruncatedSVD step on the training matrix only.

The benchmark report records shape, feature count, sparsity, zero-vectors,
elapsed time, approximate sparse-memory use, finite-value status, and identifier
order preservation.

## Planned Stage 3 — Transformers and reduction

```text
04_03_transformer_embeddings_and_reduction.ipynb
```

Planned work:

- frozen general BERT embeddings;
- frozen FinBERT embeddings and compatible sentiment probabilities;
- batch processing and resume support;
- PCA recipes for dense embeddings;
- TruncatedSVD recipes for sparse TF-IDF;
- final feature benchmark.

## Handoff

- **Phase 5** builds event-aligned market targets and joins by `source_row_id`.
- **Phase 6** owns time-aware splitting and leakage control.
- **Phase 7** joins approved features and labels, trains models, and records the
  exact representation and recipe.
