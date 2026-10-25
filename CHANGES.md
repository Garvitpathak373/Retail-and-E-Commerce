# Change Log

## Real-world dataset migration

- Replaced the synthetic customer-data workflow with independent Online Retail II and Olist pipelines.
- Added customer-level behavioral features, model comparison, stability checks, and separate artifact directories.
- Replaced synthetic-input dashboard forms with dataset-aware views, batch assignment, and aggregate-only segment advice.
- Preserved legacy source files/data as archive material only; they are no longer loaded by the primary pipeline.
