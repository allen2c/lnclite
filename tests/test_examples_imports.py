"""Example script import hygiene tests."""

import importlib
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_SCRIPTS = [
    path for path in EXAMPLES_DIR.glob("*.py") if path.name != "__init__.py"
]


def test_examples_do_not_mutate_sys_path():
    for path in EXAMPLE_SCRIPTS:
        source = path.read_text(encoding="utf-8")
        assert "sys.path" not in source, path


def test_examples_use_packaged_sample_data():
    for path in EXAMPLE_SCRIPTS:
        source = path.read_text(encoding="utf-8")
        assert "from sample_data import" not in source, path

    sample_data = importlib.import_module("lnclite.examples.sample_data")
    assert sample_data.SAMPLE_DOCUMENTS
    assert sample_data.format_document(sample_data.SAMPLE_DOCUMENTS[0])
