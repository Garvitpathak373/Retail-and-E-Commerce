# Change Log

## Quality, Reliability & Architecture Finalization (2026-10-29)

- Added deterministic conversational intent routing in `llm_advisor.py`: greetings (`hi`, `hello`, `how are you`, `thanks`) return natural conversational replies without making Gemini API calls or triggering unsolicited recommendation templates.
- Segregated substantive analytics questions with greeting prefixes (`hi, explain Recent Engaged Customers`, `hello, what is the average spend?`) to ensure verified dataset metrics and segment profiles are accurately delivered.
- Replaced technical quota-exhaustion messages (`429 RESOURCE_EXHAUSTED`, `[LOCAL FALLBACK]`) with a clean, standardized offline mode notice (`"AI insights are temporarily operating in offline mode. Recommendations are based on available dataset metrics."`).
- Added defensive validation in `process_batch_inference` (empty files, whitespace-only, >50MB limit, header-only CSVs).
- Added defensive empty-state handling in `explain_page`.
- Created comprehensive documentation suite in `docs/`: `ARCHITECTURE.md`, `METHODOLOGY.md`, `TESTING.md`, `DEPLOYMENT.md`, `CHANGELOG.md`, `ROADMAP.md`.
- Expanded automated test suite to 48 comprehensive tests covering pipeline math, AI reliability, intent routing, privacy guardrails, performance caching, and UI navigation.

## Performance Optimization & Apple UI Redesign (2026-10-28)

- Implemented Apple-inspired minimalist design system across all 6 views with custom scoped CSS, `#F5F5F7` background, `#0071E3` accents, and ligature-safe icon rendering.
- Added `@st.cache_data` caching layer across KPIs, segment distributions, population averages, and batch inference (< 50ms rerun speed).
- Fixed runtime error `TypeError: comparison_page() got an unexpected keyword argument 'dataset'`.
- Implemented lazy loading in `app.py` router to load only page-relevant artifacts.

## Clustering reproducibility upgrade (2026-10-27)

- Replaced K-Means-only deployment with deterministic comparison of K-Means, Ward, and GMM across K=3..6, using silhouette then stability tie-breaking.
- Added bounded training-only stability sampling, full-population GMM membership probabilities, cross-method ARI, collinearity logs, manifests, and method-aware batch prediction.
- Made Ward deployment memory-safe by fitting Ward on the documented sample and assigning all customers through its fitted centroids.
- Made the LLM `.env` helper optional so mock mode and pytest work even when `python-dotenv` is not installed.

## Earlier gap audit (2026-10-27)

- Model comparison selected K-Means only; no per-algorithm winner/tie-break table, bootstrap confidence intervals, cross-method ARI, or chosen-model prediction adapters.
- Olist lacked installment and delivery-delay features; collinearity removals lacked pair-level audit logging.
- Advisor used Responses API and silent mock fallback rather than Gemini-compatible Chat Completions, strict schema, safe errors, aggregate-only validation, and run logging.
- LLM evaluation, model manifest, paper assets, license/citation metadata, optional HDBSCAN diagnostics, temporal outputs, and acceptance tests were absent.
- The dashboard lacked dataset-quality, chosen-method, membership, chat, and advisor-validation views required by this specification.

## Real-world dataset migration

- Replaced the synthetic customer-data workflow with independent Online Retail II and Olist pipelines.
- Added customer-level behavioral features, model comparison, stability checks, and separate artifact directories.
- Replaced synthetic-input dashboard forms with dataset-aware views, batch assignment, and aggregate-only segment advice.
- Preserved legacy source files/data as archive material only; they are no longer loaded by the primary pipeline.
