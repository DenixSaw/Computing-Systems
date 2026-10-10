"""Накопительное сохранение результатов запусков в CSV (стандартный модуль ``csv``).

* ``runs.csv``      — одна строка = один запуск;
* ``run_nodes.csv`` — одна строка = один узел в одном запуске (связь по ``run_id``).

Файлы дописываются (append), заголовок пишется только при создании файла.
Если основной файл недоступен на запись (например, открыт в Excel), строка
сохраняется в запасной файл ``runs_<время>.csv`` / ``run_nodes_<время>.csv``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

RUNS_FILE = "runs.csv"
NODES_FILE = "run_nodes.csv"

RUNS_COLUMNS = [
    "run_id", "timestamp", "function", "N", "split_mode", "series", "nodes_count",
    "nodes_list", "total_cores", "total_time_s", "integral_value", "abs_error",
    "rel_error", "status",
]
NODES_COLUMNS = [
    "run_id", "node_name", "ip", "cores", "slice_start", "slice_end", "share_percent",
    "node_time_s", "partial_sum", "is_master",
]


@dataclass
class SaveResult:
    """Что и куда записано."""

    run_id: int
    runs_path: Path | None = None
    nodes_path: Path | None = None
    used_fallback: bool = False
    errors: list[str] = field(default_factory=list)


def _cell(value: Any) -> str:
    """Значение для CSV: None -> пусто, bool -> true/false, float -> repr (десятичная точка)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


class ResultsStore:
    """Запись результатов запусков в папку ``runs_dir``."""

    def __init__(self, runs_dir: Path | str, log: Callable[[str], None] | None = None) -> None:
        self.runs_dir = Path(runs_dir)
        self._log = log or (lambda _m: None)

    @property
    def runs_path(self) -> Path:
        return self.runs_dir / RUNS_FILE

    @property
    def nodes_path(self) -> Path:
        return self.runs_dir / NODES_FILE

    def ensure_dir(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # --- run_id -----------------------------------------------------------------

    def runs_files(self) -> list[Path]:
        """Основной runs.csv и все запасные runs_*.csv."""
        files = [self.runs_path] if self.runs_path.exists() else []
        files += sorted(self.runs_dir.glob("runs_*.csv"))
        return files

    def nodes_files(self) -> list[Path]:
        files = [self.nodes_path] if self.nodes_path.exists() else []
        files += sorted(self.runs_dir.glob("run_nodes_*.csv"))
        return files

    def next_run_id(self) -> int:
        """Следующий run_id: продолжает нумерацию после перезапуска программы.

        Берётся последняя строка каждого файла запусков (основного и запасных),
        из найденных значений выбирается максимум.
        """
        last = 0
        for path in self.runs_files():
            try:
                with path.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        try:
                            last = max(last, int(row.get("run_id") or 0))
                        except ValueError:
                            continue
            except OSError as exc:
                self._log(f"⚠ Не удалось прочитать {path.name}: {exc}")
        return last + 1

    # --- запись -------------------------------------------------------------------

    def save_run(self, run: dict[str, Any], nodes: Iterable[dict[str, Any]]) -> SaveResult:
        """Записать запуск и его узлы. ``run_id`` назначается здесь и возвращается.

        Ничего не бросает: ошибки записи попадают в журнал и в ``SaveResult.errors``.
        """
        try:
            self.ensure_dir()
        except OSError as exc:
            msg = f"✖ Не удалось создать папку результатов {self.runs_dir}: {exc}"
            self._log(msg)
            return SaveResult(run_id=0, errors=[msg])

        run_id = self.next_run_id()
        result = SaveResult(run_id=run_id)
        run_row = dict(run, run_id=run_id)
        run_row.setdefault("timestamp", datetime.now().astimezone().isoformat(timespec="seconds"))
        node_rows = [dict(n, run_id=run_id) for n in nodes]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result.runs_path = self._append_with_fallback(
            self.runs_path, f"runs_{stamp}.csv", RUNS_COLUMNS, [run_row], result)
        result.nodes_path = self._append_with_fallback(
            self.nodes_path, f"run_nodes_{stamp}.csv", NODES_COLUMNS, node_rows, result)
        return result

    def _append_with_fallback(self, path: Path, fallback_name: str, columns: list[str],
                              rows: list[dict[str, Any]], result: SaveResult) -> Path | None:
        try:
            self._append(path, columns, rows)
            return path
        except OSError as exc:
            fallback = self.runs_dir / fallback_name
            msg = (f"⚠ Файл {path.name} недоступен для записи ({exc.__class__.__name__}: {exc}). "
                   f"Возможно, он открыт в Excel. Запуск сохранён в запасной файл {fallback.name}")
            self._log(msg)
            result.errors.append(msg)
            result.used_fallback = True
            try:
                self._append(fallback, columns, rows)
                return fallback
            except OSError as exc2:
                msg2 = f"✖ Не удалось записать и запасной файл {fallback.name}: {exc2}"
                self._log(msg2)
                result.errors.append(msg2)
                return None

    @staticmethod
    def _append(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
        """Дозаписать строки; заголовок — только если файл новый или пустой."""
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            if new_file:
                writer.writerow(columns)
            for row in rows:
                writer.writerow([_cell(row.get(col)) for col in columns])
            f.flush()
            os.fsync(f.fileno())
