# lnclite

`lnclite` is a small async LanceDB document store for OpenAI-compatible embeddings.

It focuses on a compact workflow:

1. Create or load a LanceDB-backed store.
2. Add text documents with tags.
3. Build an index when the collection is large enough.
4. Search semantically with optional tag filters.

For large local ingestion workloads, `lnclite` also exposes count-returning
bulk insert APIs that avoid constructing returned document objects.

```python
from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import DocumentCreate, Lnclite, get_openai_embeddings_model
```

See [Basic Usage](usage/basic.md) for a complete runnable pattern.
See [Bulk Ingestion](usage/bulk-ingestion.md) when ingesting large corpora.
