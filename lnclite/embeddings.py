"""Embedding helpers for OpenAI-compatible embedding models."""

import functools
import logging

import diskcache
import tiktoken
from openai import AsyncOpenAI
from openai_embeddings_model import MAX_BATCH_SIZE as DEFAULT_EMBEDDINGS_MAX_BATCH_SIZE
from openai_embeddings_model import AsyncOpenAIEmbeddingsModel, ModelSettings

from lnclite.constants import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


@functools.cache
def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI()


@functools.cache
def get_embeddings_cache() -> diskcache.Cache:
    return diskcache.Cache(".cache/embeddings")


@functools.cache
def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Encoding for model %s not found, using 'gpt-5'", model)
        return tiktoken.encoding_for_model("gpt-5")


def get_openai_embeddings_model(
    model: str = DEFAULT_OPENAI_MODEL,
    openai_client: AsyncOpenAI | None = None,
    cache: diskcache.Cache | None = None,
    encoding: tiktoken.Encoding | None = None,
    max_batch_size: int = DEFAULT_EMBEDDINGS_MAX_BATCH_SIZE,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> AsyncOpenAIEmbeddingsModel:
    return AsyncOpenAIEmbeddingsModel(
        model=model,
        openai_client=openai_client or get_openai_client(),
        cache=cache or get_embeddings_cache(),
        encoding=encoding or get_encoding_for_model(model),
        max_batch_size=max_batch_size,
        max_input_tokens=max_input_tokens,
    )


def get_model_settings(dimensions: int = DEFAULT_DIMENSIONS) -> ModelSettings:
    return ModelSettings(dimensions=dimensions)
