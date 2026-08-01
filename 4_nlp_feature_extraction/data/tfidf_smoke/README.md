# TF-IDF smoke-test artifacts

Run `notebooks/04_02_tfidf_feature_extraction.ipynb` to generate:

- six sparse `.npz` matrices;
- feature-name JSON files;
- fitted smoke-test vectorizer Joblib files;
- `smoke_sample_source_row_ids.csv` preserving exact matrix row order.

These artifacts are diagnostic only. Final TF-IDF vectorizers must be fit on the
training split after the temporal split is defined.
