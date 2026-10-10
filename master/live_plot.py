"""Живой график расчёта в окне Master: профиль P(x) = ∫∫ f(x, y, z) dy dz по слоям X.

Каждый слой красится цветом узла, который его посчитал; светлым фоном показаны
доли узлов. Площадь под кривой — это интеграл; накопленное значение
(hx/3)·Σ wi·P(xi) по готовым слоям в точности равно сумме уже полученных вкладов.
Перерисовка — по таймеру не чаще 4 раз в секунду, поэтому окно не тормозит.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from common.integrator import TaskParams, simpson_weights
from master.server import MASTER_NODE_ID, Assignment

MASTER_COLOR = "#ff7f0e"   # как на nodes_time.png
SLAVE_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
REDRAW_MS = 250


class LiveProfilePlot(QWidget):
    """Виджет с живым профилем расчёта."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.info = QLabel("Запустите расчёт — здесь в реальном времени появится профиль "
                           "∫∫ f dy dz по слоям X, раскрашенный по узлам.")
        self.info.setWordWrap(True)
        self.figure = Figure(figsize=(7, 3.6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(1, 1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.info)
        layout.addWidget(self.canvas, 1)

        self.params: TaskParams | None = None
        self.x = np.zeros(0)
        self.w = np.zeros(0)
        self.profile = np.zeros(0)
        self.owner = np.zeros(0, dtype=int)
        self.nodes: dict[int, tuple[str, str]] = {}     # node_id -> (имя, цвет)
        self._dirty = False
        self._final_text = ""

        self._timer = QTimer(self)
        self._timer.setInterval(REDRAW_MS)
        self._timer.timeout.connect(self._redraw_if_dirty)
        self._timer.start()
        self._draw_empty()

    # --- управление -----------------------------------------------------------

    def reset(self, params: TaskParams, assignments: Iterable[Assignment]) -> None:
        """Новый расчёт: сетка по X, доли узлов, цвета."""
        self.params = params
        self.x = params.ax + params.hx * np.arange(params.n + 1)
        self.w = simpson_weights(params.n)
        self.profile = np.full(params.n + 1, np.nan)
        self.owner = np.full(params.n + 1, -1, dtype=int)
        self.assignments = list(assignments)
        self.nodes = {}
        k = 0
        for a in self.assignments:
            if a.node_id == MASTER_NODE_ID:
                color = MASTER_COLOR
            else:
                color = SLAVE_COLORS[k % len(SLAVE_COLORS)]
                k += 1
            self.nodes[a.node_id] = (a.name, color)
        self._final_text = ""
        self._dirty = True
        self._redraw()

    def add_chunks(self, node_id: int, chunks: list) -> None:
        """Порции от узла: [[первый слой, профиль, вклад], …]."""
        if self.params is None:
            return
        n1 = self.params.n + 1
        for chunk in chunks:
            try:
                start, values = int(chunk[0]), chunk[1]
            except (IndexError, TypeError, ValueError):
                continue
            end = min(start + len(values), n1)
            if start < 0 or end <= start:
                continue
            self.profile[start:end] = np.asarray(values[: end - start], dtype=float)
            self.owner[start:end] = node_id
        self._dirty = True

    def finish(self, integral: float | None, status: str) -> None:
        """Расчёт завершён: финальная подпись и перерисовка."""
        if integral is not None:
            self._final_text = f"Готово: I = {integral:.15g}"
        else:
            self._final_text = {"stopped": "Расчёт остановлен",
                                "failed": "Расчёт завершился ошибкой"}.get(status, status)
        self._dirty = True
        self._redraw()

    # --- отрисовка ----------------------------------------------------------------

    def _accumulated(self) -> tuple[float, float]:
        """(доля готовых слоёв, накопленный интеграл по готовым слоям)."""
        assert self.params is not None
        done = ~np.isnan(self.profile)
        acc = self.params.hx / 3.0 * float(np.sum(self.w[done] * self.profile[done]))
        return float(done.mean()), acc

    def _draw_empty(self) -> None:
        self.ax.clear()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("∫∫ f dy dz")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _redraw_if_dirty(self) -> None:
        if self._dirty and self.isVisible():
            self._redraw()

    def _redraw(self) -> None:
        self._dirty = False
        if self.params is None:
            return
        p = self.params
        ax = self.ax
        ax.clear()
        ax.grid(True, alpha=0.3)

        # Светлый фон: какая доля слоёв назначена какому узлу
        for a in self.assignments:
            name, color = self.nodes[a.node_id]
            x0, x1 = self.x[a.start], self.x[a.end - 1]
            ax.axvspan(x0, x1 if x1 > x0 else x0 + p.hx, color=color, alpha=0.08, linewidth=0)

        # Кривая: отдельно для каждого узла — только посчитанные им слои
        has_data = False
        for node_id, (name, color) in self.nodes.items():
            mask = self.owner == node_id
            if not mask.any():
                ax.plot([], [], color=color, linewidth=2, label=name)
                continue
            has_data = True
            # Захватываем соседнюю слева готовую точку, чтобы на стыке узлов не было разрыва
            ext = mask.copy()
            ext[:-1] |= mask[1:] & ~np.isnan(self.profile[:-1])
            y = np.where(ext, self.profile, np.nan)
            ax.plot(self.x, y, color=color, linewidth=2, label=name)
            ax.fill_between(self.x, 0, y, where=ext, color=color, alpha=0.25, linewidth=0)

        ax.set_xlim(p.ax, p.bx)
        if has_data:
            lo = min(0.0, float(np.nanmin(self.profile)))
            hi = max(0.0, float(np.nanmax(self.profile)))
            pad = (hi - lo) * 0.08 or 1.0
            ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("∫∫ f dy dz")
        ax.set_title(f"Профиль по слоям X: функция {p.function}, N = {p.n}", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        frac, acc = self._accumulated()
        text = f"Посчитано слоёв: {frac * 100:.0f} %   ·   накоплено I ≈ {acc:.12g}"
        if self._final_text:
            text = f"{self._final_text}   ·   {text}"
        self.info.setText(text)
