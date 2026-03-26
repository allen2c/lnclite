import os
from pathlib import Path

import xxhash


def get_folder_fingerprint(target_path: Path | str, read_content: bool = False) -> str:
    # Build a single rolling hash with xxh64.
    folder_hash = xxhash.xxh64()

    # Keep traversal order stable; otherwise the same tree can hash differently
    # if os.walk yields files in a different order.
    for root, dirs, files in os.walk(target_path):
        for names in sorted(files):
            file_path = os.path.join(root, names)

            try:
                # 1. Mix in relative path (detects moves/renames).
                rel_path = os.path.relpath(file_path, target_path)
                folder_hash.update(rel_path.encode())

                # 2. Mix in file metadata — fast default.
                # Use content hashing only if you need stronger accuracy
                # and can pay the I/O cost.
                if read_content:
                    # Catches changes where mtime/size might not move (rare edge cases).
                    with open(file_path, "rb") as f:
                        folder_hash.update(f.read())
                else:
                    stat = os.stat(file_path)
                    folder_hash.update(str(stat.st_mtime).encode())  # modification time
                    folder_hash.update(str(stat.st_size).encode())  # file size

            except (PermissionError, FileNotFoundError):
                continue

    return folder_hash.hexdigest()
