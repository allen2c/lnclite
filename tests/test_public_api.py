"""Public API tests."""

import lnclite


def test_public_api_exports_release_surface():
    assert lnclite.__all__ == [
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


def test_exported_names_are_importable():
    for name in lnclite.__all__:
        assert getattr(lnclite, name) is not None
