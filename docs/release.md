# Release

Run these checks before manually building and publishing:

```bash
python -m pytest -q
poetry check
poetry build
twine check dist/*
```

Inspect the built distributions before publishing:

```bash
python -m tarfile -l dist/*.tar.gz
python -m zipfile -l dist/*.whl
```

The repository does not include PyPI publish automation.
