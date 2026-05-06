# Managing Multiple Stores

`LncliteManager` helps applications manage many local datasets without keeping
every client open.

```python
from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import LncliteConfig, LncliteManager, get_openai_embeddings_model


embeddings = get_openai_embeddings_model(
    openai_client=AsyncOpenAI(),
)

manager = LncliteManager(max_clients=16, idle_ttl_seconds=300)

client = await manager.open(
    "dataset-a",
    config=LncliteConfig(
        lancedb_path="/var/app/cache/dataset-a.lance",
        openai_embeddings_model=embeddings,
        model_settings=ModelSettings(dimensions=1536),
        vector_search_prefer="storage",
    ),
)
```

Use `create=True` when the manager should create a new store:

```python
client = await manager.open(
    "dataset-a",
    config=config,
    create=True,
)
```

Cleanup helpers:

```python
await manager.close("dataset-a")
await manager.close_idle()
await manager.close_all()
```

`lnclite` does not provide HTTP, authentication, authorization, or routing.
Applications should implement those concerns outside the library.
