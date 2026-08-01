# Full frozen embedding artifacts

Local full-dataset BERT/FinBERT jobs write float32 `.npy` memmaps, shared
`source_row_id` mappings, progress files, and metadata here. Batch processing and
resume are supported. The generated arrays are intentionally excluded from Git
and delivery ZIP files because they are large and reproducible.
