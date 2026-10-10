"""Мелкие помощники для PyQt (не импортируются процессами пула)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QPlainTextEdit


class CallThread(QThread):
    """Выполнить функцию вне GUI-потока и вернуть результат сигналом ``done``."""

    done = pyqtSignal(object)      # результат функции
    failed = pyqtSignal(str)       # текст ошибки

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn())
        except Exception as exc:  # noqa: BLE001 — ошибку показываем в журнале
            self.failed.emit(repr(exc))


def append_log(widget: QPlainTextEdit, text: str) -> None:
    """Добавить строку журнала с меткой времени и прокрутить вниз."""
    widget.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
    sb = widget.verticalScrollBar()
    sb.setValue(sb.maximum())
