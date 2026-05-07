# lnclite Examples

These examples are runnable scripts for the current `lnclite` API. They use
`openai.AsyncOpenAI()` directly, so configure your OpenAI-compatible embedding
environment before running them.

If `lnclite` is installed in your Python environment:

```bash
/Users/allenchou/miniconda3/envs/lnclite/bin/python examples/basic_semantic_search.py
```

For local source-tree runs without an editable install, run from the repository
root and prefix the command with `PYTHONPATH=.`:

```bash
PYTHONPATH=. /Users/allenchou/miniconda3/envs/lnclite/bin/python examples/basic_semantic_search.py
```

## Scripts

- `basic_semantic_search.py`: create a local store, add bundled documents,
  create an index, and run semantic search.
- `tagged_search_and_listing.py`: search with `tags_any`, list with `tags_all`,
  and fetch a second page with a pagination token.
- `directory_ingest.py`: generate markdown files, ingest a directory, search,
  and filter by the generated `path:...` tag.
- `manage_multiple_stores.py`: manage two stores with `LncliteManager`, show
  cache eviction, reopen a store, and clean up.
- `open_dataset_search.py`: download the small Palmer Penguins CSV dataset,
  convert rows to natural-language documents, and search them.

Generated LanceDB stores are written under `outputs/examples/`.

## Optional Dataset Demo

`open_dataset_search.py` downloads:

```text
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
```

The CSV is cached at `outputs/examples/data/penguins.csv` for repeat runs. The
Palmer Penguins teaching dataset is commonly distributed as CC0 public-domain
data.
