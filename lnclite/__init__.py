"""Small async LanceDB document store with OpenAI-compatible embeddings."""

from typing import Final

from lnclite.client import Lnclite
from lnclite.config import LncliteConfig
from lnclite.embeddings import get_model_settings, get_openai_embeddings_model
from lnclite.indexing import DocumentIndexPlan
from lnclite.manager import LncliteManager
from lnclite.models import (
    Document,
    DocumentCreate,
    LncliteNotFoundError,
    ManifestModel,
    SearchResult,
    SearchResults,
)

__version__: Final[str] = "0.2.0"
__all__: Final[list[str]] = [
    "Document",
    "DocumentCreate",
    "DocumentIndexPlan",
    "Lnclite",
    "LncliteConfig",
    "LncliteManager",
    "LncliteNotFoundError",
    "ManifestModel",
    "SearchResult",
    "SearchResults",
    "get_model_settings",
    "get_openai_embeddings_model",
]
