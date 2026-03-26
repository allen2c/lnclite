from pathlib import Path

import xxhash


def get_file_fp(file_path: Path | str) -> str:
    hasher = xxhash.xxh64()

    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    return hasher.hexdigest()
