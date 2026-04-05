from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class AppendSafeCsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        needs_header = (not file_exists) or self.path.stat().st_size == 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames, extrasaction="ignore")
        if needs_header:
            self._writer.writeheader()
            self._handle.flush()

    def write_row(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()

    def __enter__(self) -> "AppendSafeCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
