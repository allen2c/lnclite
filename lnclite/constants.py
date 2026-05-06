"""Shared constants and type aliases for lnclite."""

from typing import Literal

DEFAULT_MANIFEST_TABLE = "manifest"
DEFAULT_DOCUMENT_TABLE = "documents"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_DIMENSIONS = 1536

VectorIndexPreference = Literal["storage", "balanced", "accuracy", "latency"]
ListOrder = Literal["asc", "desc", 1, -1]
SqlOrder = Literal["ASC", "DESC"]
