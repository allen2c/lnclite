# Installation

Install from PyPI:

```bash
pip install lnclite
```

For local development:

```bash
poetry install --all-groups
```

## Environment

By default, `AsyncOpenAI()` reads credentials from the environment:

```bash
export OPENAI_API_KEY="..."
```

You can also pass an OpenAI-compatible endpoint:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://example.test/v1",
    api_key="example-key",
)
```
