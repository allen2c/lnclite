from pathlib import Path
from typing import Final, Text

__version__: Final[Text] = "0.1.0"


class Lnclite:
    def __init__(self, files_dir: Path | str = "."):
        self.files_dir = Path(files_dir).resolve()
        if not self.files_dir.is_dir():
            raise FileNotFoundError(f"Files directory {self.files_dir} not found")

        self.lancedb_path = self.files_dir.parent.joinpath(
            "." + self.files_dir.name + ".index"
        )

        self.last_fingerprint = None

    def search(self):
        pass

    def sync(self):
        pass

    def is_synced(self) -> bool:
        if self.last_fingerprint is None:
            from lnclite.utils.get_folder_fingerprint import get_folder_fingerprint

            self.last_fingerprint = get_folder_fingerprint(self.files_dir)

        return self.last_fingerprint == get_folder_fingerprint(self.files_dir)
