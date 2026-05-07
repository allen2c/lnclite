"""Shared sample data for lnclite examples."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

SAMPLE_DOCUMENTS: list[_SampleDocument] = [
    {
        "title": "Python async tasks",
        "content": (
            "Python async code uses coroutines and an event loop to handle "
            "concurrent I/O without blocking a thread for every operation."
        ),
        "tags": ["type:guide", "topic:python", "level:intro"],
    },
    {
        "title": "SQLite local storage",
        "content": (
            "SQLite stores relational data in a single local file and works "
            "well for small applications, tests, and embedded tooling."
        ),
        "tags": ["type:guide", "topic:database", "level:intro"],
    },
    {
        "title": "HTTP request methods",
        "content": (
            "HTTP clients commonly use GET to read resources, POST to create "
            "resources, PATCH to update part of a resource, and DELETE to "
            "remove one."
        ),
        "tags": ["type:reference", "topic:http", "level:intro"],
    },
    {
        "title": "Git feature branches",
        "content": (
            "Git feature branches isolate work until commits are reviewed and "
            "merged back into the main integration branch."
        ),
        "tags": ["type:guide", "topic:git", "level:intro"],
    },
    {
        "title": "Markdown documentation",
        "content": (
            "Markdown keeps documentation readable in source form while still "
            "supporting headings, links, lists, and code blocks."
        ),
        "tags": ["type:guide", "topic:docs", "level:intro"],
    },
    {
        "title": "Release verification",
        "content": (
            "Release verification checks tests, package metadata, built "
            "artifacts, and documentation before publishing a version."
        ),
        "tags": ["type:checklist", "topic:release", "level:intermediate"],
    },
]


def format_document(document: _SampleDocument) -> str:
    return f"{document['title']}\n\n{document['content']}"


def document_tags(document: _SampleDocument) -> list[str]:
    return list(document["tags"])


def write_sample_notes(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for document in SAMPLE_DOCUMENTS:
        title = document["title"]
        content = document["content"]
        filename = _slugify(title) + ".md"
        path = directory / filename
        path.write_text(
            f"# {title}\n\n{content}\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "-")


class _SampleDocument(TypedDict):
    title: str
    content: str
    tags: list[str]
