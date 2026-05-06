# lnclite

`lnclite` is a small async LanceDB document store for OpenAI-compatible embeddings.

It focuses on a compact workflow:

1. Create or load a LanceDB-backed store.
2. Add text documents with tags.
3. Build an index when the collection is large enough.
4. Search semantically with optional tag filters.

```python
from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import DocumentCreate, Lnclite, get_openai_embeddings_model
```

See [Basic Usage](usage/basic.md) for a complete runnable pattern.
