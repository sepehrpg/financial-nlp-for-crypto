# Phase 2 — Data Audit and Exploratory Data Analysis

This folder contains the second phase of the Financial NLP for Crypto project.
The phase is intentionally notebook-centered because the current goal is
learning, research, and transparent exploratory analysis rather than production
orchestration.

## Purpose

Before preprocessing text or training any NLP model, we need to answer a more
basic question:

> **What is actually inside the dataset, how reliable is it, and what risks must
> be handled before modeling?**

This phase therefore audits the raw historical dataset without modifying it.

## Relationship with Phase 1

The existing acquisition phase has the following structure:

```text
1_data_acquisition/
└── historical/
    ├── download_datasets.py
    ├── inspect_dataset.py
    └── row/
        ├── cryptovision_v1.csv
        └── cryptovision_v2.csv
```

`inspect_dataset.py` is intentionally lightweight. It checks whether downloaded
files can be opened and prints a small sample-based summary.

Phase 2 goes further. It performs research-oriented checks such as:

- documented schema vs observed schema,
- full-dataset missingness,
- duplicate article identities,
- timestamp validity and temporal coverage,
- cryptocurrency/category coverage,
- source-domain imbalance,
- text completeness, length, and noise,
- sentiment-label and market-move distributions,
- OHLCV sanity checks,
- leakage review for the initial text-only modeling setup.

The two phases therefore have different responsibilities and should both exist.

## Folder Structure

```text
2_data_audit_and_eda/
├── README.md
├── notebooks/
│   └── 02_data_audit_and_eda.ipynb
└── src/
    ├── __init__.py
    ├── audit_utils.py
    └── eda_utils.py
```

There is intentionally **no `run_audit.py`** and no report-generation pipeline.
The notebook is the main executable artifact for this student/research project.

## Dataset Used in the Current Project

The initial modeling dataset is CryptoVision V1:

- DOI: `10.17632/wvjjxr8bxx.1`
- Mendeley dataset version: V1
- Main published file: `CryptoDataSet.csv`
- Published coverage: cryptocurrency news from 2017–2025

In this repository, Phase 1 downloads it using the local filename:

```text
cryptovision_v1.csv
```

CryptoVision V2 may also exist in the acquisition folder. It is treated as an
optional comparison dataset, not as a silent replacement for V1, because V2 has
a different published structure and preprocessing history.

## Why the Notebook Is the Main Interface

For this phase, the notebook is useful because each analysis can be presented as:

1. **Question** — what are we trying to learn?
2. **Reason** — why does it matter for Financial NLP?
3. **Code** — how is the check performed?
4. **Result** — what did the dataset show?
5. **Interpretation** — what does the result mean?
6. **Decision** — what should Phase 3 do differently because of this finding?

That format is more educational and scientifically transparent than hiding the
whole process behind a command-line script.

## Memory Strategy

CryptoVision V1 is large enough that loading the full CSV into memory may be
unnecessary or inconvenient on a student laptop.

The code therefore uses two complementary strategies:

### Small sample

A small sample is used for fast inspection and visual/text diagnostics.

### Streaming full-data checks

Checks that should represent the whole dataset are calculated chunk by chunk,
including:

- row count,
- missingness,
- category counts,
- source-domain counts,
- duplicate URL counts,
- timestamp validity,
- OHLCV consistency checks.

This keeps memory usage bounded while still auditing all rows.

## Important Methodological Rule: Do Not Trust Existing Labels Blindly

CryptoVision V1 already contains columns such as:

```text
Sentiment_Label
Sentiment_Score
Open
High
Low
Close
Volume
Movement_OpenClose_%
Movement_HighLow_%
Market_Move
```

They are useful for auditing and understanding the dataset, but they should not
be automatically treated as features or as the final target definitions for this
project.

The initial model is defined as **text-only**. Therefore market outcomes and
publisher-generated sentiment annotations must not enter its feature matrix.

The project also intends to create its own market-impact labels for horizons such
as 1 hour, 4 hours, and 24 hours. That work belongs to Phase 5:
`5_market_alignment_and_labeling`.

## Running the Notebook

Start Jupyter from the repository root or from the Phase 2 folder and open:

```text
2_data_audit_and_eda/notebooks/02_data_audit_and_eda.ipynb
```

The notebook automatically searches upward for the repository root by locating
`1_data_acquisition`.

The current acquisition layout uses:

```text
1_data_acquisition/historical/row/
```

The helper code also supports a future rename from `row` to the more conventional
`raw` without requiring notebook changes.

## Python Dependencies

The notebook uses standard data-science packages:

```text
pandas
numpy
matplotlib
jupyter
```

No additional NLP model is required in Phase 2.

## Main Questions Answered by This Phase

By the end of the notebook, you should be able to answer:

- Does the downloaded V1 file match its documented schema?
- How many usable records exist?
- Which columns are incomplete?
- Are article URLs duplicated?
- Is publication time parseable and what is the actual date range?
- How much of the dataset is relevant to Bitcoin?
- Are a few publishers dominating the dataset?
- Which text field is most suitable for the next phase?
- How noisy are Title, Description, Full_Text, and Filtered_Text?
- Are sentiment labels or market classes severely imbalanced?
- Are OHLCV values internally plausible?
- Which columns would create target leakage in a text-only model?
- What concrete preprocessing decisions should be carried into Phase 3?

## Expected Phase 2 Output

The primary output is **knowledge about the dataset**, not a new transformed CSV.
Raw files must remain unchanged.

At the end of the notebook, write a short decision summary covering:

- selected dataset version,
- selected text representation candidate,
- columns to keep for later phases,
- columns to exclude from initial model features,
- missing-value issues,
- duplicate policy to test in preprocessing,
- source/time imbalance concerns,
- known limitations that must be carried forward.

Those decisions form the input contract for Phase 3: `3_text_preprocessing`.
