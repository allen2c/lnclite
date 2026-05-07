# v0.2.0 Examples Design

## Goal

Enrich `examples/` with runnable end-to-end scripts that demonstrate the current
`lnclite` API without changing library behavior for `v0.2.0`.

## Scope

Add example scripts only. Do not add document update/delete, chunking, CLI
commands, deduplication behavior, or runtime API changes.

## Example Set

The examples will have two tiers:

1. Core bundled-data examples that run without downloading any dataset.
2. One optional real-dataset example that downloads and caches a small open
   dataset for a more realistic demo.

Core scripts:

- `examples/basic_semantic_search.py`
- `examples/tagged_search_and_listing.py`
- `examples/directory_ingest.py`
- `examples/manage_multiple_stores.py`

Optional real-dataset script:

- `examples/open_dataset_search.py`

Shared bundled data:

- `examples/sample_data.py`

Documentation:

- `examples/README.md`

## Data Design

Bundled examples use a small repo-owned technical-notes dataset in
`examples/sample_data.py`. The notes cover simple topics such as Python async,
SQLite, HTTP, Git, Markdown, testing, and releases. The data is embedded in the
repository to keep the default examples reliable and easy to inspect.

The optional real-dataset example downloads a small open text or CSV dataset
with a license/source comment in the script. It caches files under
`outputs/examples/data/` so repeated runs do not download again.

## Runtime Design

Each script is runnable with:

```bash
/Users/allenchou/miniconda3/envs/lnclite/bin/python examples/<script>.py
```

The scripts use:

- `openai.AsyncOpenAI()` directly.
- `get_openai_embeddings_model(openai_client=AsyncOpenAI())`.
- `ModelSettings(dimensions=1536)`.
- Output paths under `outputs/examples/`.

Each script may delete only its own generated output directory before creating a
fresh LanceDB store.

## Error Handling

Examples should fail plainly if embedding credentials or an OpenAI-compatible
endpoint are not configured. The optional downloaded dataset example should
raise a clear error if download fails.

## Testing

Live examples should not run as part of normal `pytest` because they call an
embedding API. Verification for this change is:

- Run the existing test suite.
- Run syntax compilation for `examples/`.
- Manually run at least one core example with the configured conda Python.

## Non-Goals

- No mocked embedding implementation.
- No extra example-only dependencies.
- No large datasets committed to the repository.
- No changes to public library behavior.
