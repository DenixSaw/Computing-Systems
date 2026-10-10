"""Численное интегрирование составной кубатурной формулой Симпсона на сетке N×N×N.

I ≈ (hx·hy·hz / 27) · Σ wi·wj·wk · f(xi, yj, zk),  веса по оси: 1, 4, 2, 4, …, 2, 4, 1.

Сумма раскладывается по слоям X: I = (h/27) · Σ_i wi · S_i, где S_i — взвешенная
сумма по (y, z) в слое x = xi. Поэтому задачу можно делить между узлами на
непересекающиеся диапазоны слоёв [start, end): каждый узел берёт ГЛОБАЛЬНЫЕ веса wi
своих слоёв, и сумма частичных результатов в точности равна последовательной сумме
(с точностью до порядка округления).

Модуль не импортирует PyQt: он загружается в процессы пула (spawn).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np

from common.config import CHUNK_POINTS


# ---------------------------------------------------------------------------
# Параметры задачи
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskParams:
    """Параметры интеграла: функция, N (чётное) и границы параллелепипеда."""

    function: str          # "test" | "heavy"
    n: int                 # число интервалов по каждой оси (чётное)
    ax: float
    bx: float
    ay: float
    by: float
    az: float
    bz: float

    @property
    def hx(self) -> float:
        return (self.bx - self.ax) / self.n

    @property
    def hy(self) -> float:
        return (self.by - self.ay) / self.n

    @property
    def hz(self) -> float:
        return (self.bz - self.az) / self.n

    @property
    def scale(self) -> float:
        """Множитель hx·hy·hz / 27 перед суммой."""
        return self.hx * self.hy * self.hz / 27.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TaskParams":
        return TaskParams(
            function=str(d["function"]), n=int(d["n"]),
            ax=float(d["ax"]), bx=float(d["bx"]),
            ay=float(d["ay"]), by=float(d["by"]),
            az=float(d["az"]), bz=float(d["bz"]),
        )


def validate_params(p: TaskParams) -> str | None:
    """Проверить параметры. Возвращает текст ошибки или None, если всё в порядке."""
    if p.function not in ("test", "heavy"):
        return f"Неизвестная функция: {p.function}"
    if p.n < 2 or p.n % 2:
        return "N должно быть чётным и не меньше 2"
    if not (p.bx > p.ax and p.by > p.ay and p.bz > p.az):
        return "Для каждой оси верхняя граница должна быть больше нижней"
    if p.function == "heavy" and min(p.ax, p.ay, p.az) < 0:
        # ln(1 + xyz) определён при xyz > -1; на неотрицательной области — гарантированно
        return "Для тяжёлой функции область должна быть неотрицательной (все границы ≥ 0)"
    return None


# ---------------------------------------------------------------------------
# Подынтегральные функции (векторизованы, работают с массивами NumPy)
# ---------------------------------------------------------------------------

def f_test(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Тестовая функция sin(x)·cos(y)·exp(z) — у неё известен точный интеграл."""
    return np.sin(x) * np.cos(y) * np.exp(z)


def f_heavy(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Дорогая функция для замеров времени (на неотрицательной области без NaN)."""
    xy = x * y
    return (np.sin(xy) * np.exp(-z)
            + np.sqrt(x * x + y * y + z * z + 1.0) * np.cos(x + y * z)
            + np.log1p(xy * z))


FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "test": f_test,
    "heavy": f_heavy,
}


def exact_test_integral(ax: float, bx: float, ay: float, by: float,
                        az: float, bz: float) -> float:
    """Точное значение ∫∫∫ sin(x)·cos(y)·exp(z) по параллелепипеду."""
    return ((math.cos(ax) - math.cos(bx))
            * (math.sin(by) - math.sin(ay))
            * (math.exp(bz) - math.exp(az)))


# ---------------------------------------------------------------------------
# Веса Симпсона и деление слоёв между узлами
# ---------------------------------------------------------------------------

def simpson_weights(n: int) -> np.ndarray:
    """Веса 1, 4, 2, 4, …, 2, 4, 1 для n интервалов (n+1 узел)."""
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w


def partition_layers(n: int, weights: Sequence[float]) -> list[tuple[int, int]]:
    """Разбить слои X с индексами 0..n между узлами пропорционально весам.

    Возвращает список полуинтервалов [start, end) в порядке узлов. Границы
    выровнены по панелям Симпсона (пара интервалов [2k, 2k+2]), т.е. каждая
    граница — чётный индекс; последний непустой диапазон включает слой n.
    Диапазоны не пересекаются и покрывают все n+1 слоёв. Узел может получить
    пустой диапазон (start == end), если панелей меньше, чем узлов.
    """
    if n < 2 or n % 2:
        raise ValueError("n должно быть чётным и не меньше 2")
    if not weights:
        raise ValueError("Нужен хотя бы один узел")
    w = [max(0.0, float(x)) for x in weights]
    total = sum(w)
    if total <= 0:
        w = [1.0] * len(w)
        total = float(len(w))

    panels = n // 2
    quotas = [panels * x / total for x in w]
    counts = [int(math.floor(q)) for q in quotas]
    # Метод наибольшего остатка: раздаём оставшиеся панели узлам с наибольшей дробной частью
    rest = panels - sum(counts)
    order = sorted(range(len(w)), key=lambda i: (quotas[i] - counts[i], w[i]), reverse=True)
    for i in order[:rest]:
        counts[i] += 1

    ranges: list[tuple[int, int]] = []
    p = 0
    for c in counts:
        ranges.append((2 * p, 2 * (p + c)))
        p += c
    # Слой n (правая граница) отдаём последнему непустому диапазону,
    # пустые диапазоны после него сдвигаем в конец (n+1, n+1)
    for i in range(len(ranges) - 1, -1, -1):
        s, e = ranges[i]
        if e > s:
            ranges[i] = (s, n + 1)
            break
        ranges[i] = (n + 1, n + 1)
    return ranges


def chunk_layers(n: int) -> int:
    """Сколько слоёв считать за один вызов NumPy, чтобы не превысить CHUNK_POINTS точек."""
    return max(1, CHUNK_POINTS // ((n + 1) ** 2))


def split_range(start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Разрезать [start, end) на куски не длиннее size."""
    return [(s, min(s + size, end)) for s in range(start, end, size)]


# ---------------------------------------------------------------------------
# Расчёт порции слоёв (выполняется в процессе пула)
# ---------------------------------------------------------------------------

# Кэш сетки (Y, Z, Wyz) внутри процесса: строится один раз на задачу
_grid_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _grid(p: TaskParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Сетка (Y, Z) слоя, матрица весов wy⊗wz и веса по X — с кэшированием."""
    key = (p.n, p.ay, p.by, p.az, p.bz)
    cached = _grid_cache.get(key)
    if cached is None:
        _grid_cache.clear()  # держим в памяти только сетку текущей задачи
        w = simpson_weights(p.n)
        y = p.ay + p.hy * np.arange(p.n + 1)
        z = p.az + p.hz * np.arange(p.n + 1)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        cached = (yy[np.newaxis], zz[np.newaxis], np.outer(w, w), w)
        _grid_cache[key] = cached
    return cached


def compute_chunk(params: dict, start: int,
                  end: int) -> tuple[int, int, float, list[float]]:
    """Невзвешенная по h сумма Σ_{i∈[start,end)} wi · Σ_jk wj·wk·f для порции слоёв.

    Возвращает (start, end, сумма, профиль). Профиль — значения
    P(x_i) = (hy·hz/9)·S_i ≈ ∬ f(x_i, y, z) dy dz для каждого слоя порции
    (для живого графика). Вызывается в процессе пула, поэтому параметры
    передаются словарём (легко сериализуется).
    """
    p = TaskParams.from_dict(params)
    yy, zz, wyz, w = _grid(p)
    f = FUNCTIONS[p.function]
    total = 0.0
    profile: list[float] = []
    yz_scale = p.hy * p.hz / 9.0
    step = chunk_layers(p.n)
    for s, e in split_range(start, end, step):
        x = (p.ax + p.hx * np.arange(s, e))[:, np.newaxis, np.newaxis]
        values = f(x, yy, zz)                                # (e-s, n+1, n+1)
        layer_sums = np.einsum("kij,ij->k", values, wyz)     # S_i для каждого слоя
        total += float(np.dot(w[s:e], layer_sums))
        profile.extend((layer_sums * yz_scale).tolist())
    return start, end, total, profile


def compute_range(p: TaskParams, start: int, end: int) -> float:
    """Последовательный (в одном процессе) расчёт вклада слоёв [start, end) в интеграл."""
    if end <= start:
        return 0.0
    return compute_chunk(p.to_dict(), start, end)[2] * p.scale


def integrate_sequential(p: TaskParams) -> float:
    """Полный интеграл в одном процессе (эталон для проверки)."""
    return compute_range(p, 0, p.n + 1)


# ---------------------------------------------------------------------------
# Пул процессов узла
# ---------------------------------------------------------------------------

class ComputeCancelled(Exception):
    """Расчёт отменён пользователем или master."""


class ComputePool:
    """Пул процессов, загружающий все ядра узла (потоки не подходят из-за GIL).

    Пул создаётся лениво и переиспользуется между запусками, чтобы не тратить
    время на старт процессов. Отмена — через ``cancel_event``: пул убивается
    (terminate) и при следующем расчёте создаётся заново.
    """

    def __init__(self, processes: int) -> None:
        self._processes = max(1, int(processes))
        self._pool: mp.pool.Pool | None = None
        self._lock = threading.Lock()
        # spawn одинаково работает на Windows / Linux / macOS и безопасен рядом с потоками Qt
        self._ctx = mp.get_context("spawn")

    @property
    def processes(self) -> int:
        return self._processes

    def set_processes(self, processes: int) -> None:
        """Сменить число процессов (пул пересоздастся при следующем расчёте)."""
        processes = max(1, int(processes))
        with self._lock:
            if processes != self._processes:
                self._processes = processes
                self._terminate_locked()

    def warm_up(self) -> None:
        """Заранее запустить процессы пула и импортировать в них NumPy.

        Старт процессов (особенно spawn на Windows/macOS) занимает время, которое
        не должно попадать в замер первого расчёта. Блокирующий вызов — не из GUI-потока.
        """
        with self._lock:
            pool = self._ensure_locked()
        try:
            pool.map(_warm_up_task, range(self._processes))
        except Exception:  # noqa: BLE001 — пул могли закрыть во время прогрева
            pass

    def _ensure_locked(self) -> "mp.pool.Pool":
        if self._pool is None:
            self._pool = self._ctx.Pool(processes=self._processes)
        return self._pool

    def _terminate_locked(self) -> None:
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:  # noqa: BLE001 — при выходе любые ошибки пула не важны
                pass
            self._pool = None

    def compute(self, p: TaskParams, start: int, end: int,
                progress_cb: Callable[[float], None] | None = None,
                cancel_event: threading.Event | None = None,
                chunk_cb: Callable[[int, list[float], float], None] | None = None,
                ) -> tuple[float, float]:
        """Посчитать вклад слоёв [start, end) всеми процессами пула.

        Возвращает (частичный интеграл, время расчёта в секундах).
        ``progress_cb`` получает долю выполненного (0..1).
        ``chunk_cb(start, profile, partial)`` вызывается на каждую готовую порцию
        (до ``progress_cb``): первый слой порции, профиль ∬f dy dz по её слоям и
        вклад порции в интеграл — для живого графика.
        """
        if end <= start:
            return 0.0, 0.0
        with self._lock:
            fresh = self._pool is None
            pool = self._ensure_locked()
        if fresh:
            # Только что созданный пул: дождаться старта процессов до начала замера,
            # чтобы время узла включало только вычисления
            pool.map(_warm_up_task, range(self._processes))

        count = end - start
        # Куски: не больше лимита памяти и примерно 4 куска на процесс для балансировки
        size = max(1, min(chunk_layers(p.n), math.ceil(count / (self._processes * 4))))
        pieces = split_range(start, end, size)
        params = p.to_dict()

        t0 = time.perf_counter()
        results: dict[int, float] = {}
        done_layers = 0
        it = pool.imap_unordered(_compute_chunk_star, [(params, s, e) for s, e in pieces])
        while len(results) < len(pieces):
            if cancel_event is not None and cancel_event.is_set():
                with self._lock:
                    self._terminate_locked()
                raise ComputeCancelled()
            try:
                s, e, value, profile = it.next(timeout=0.2)
            except mp.TimeoutError:
                continue
            results[s] = value
            done_layers += e - s
            if chunk_cb is not None:
                chunk_cb(s, profile, value * p.scale)
            if progress_cb is not None:
                progress_cb(done_layers / count)
        elapsed = time.perf_counter() - t0
        # Суммируем в порядке слоёв — результат не зависит от порядка завершения кусков
        partial = math.fsum(results[s] for s in sorted(results)) * p.scale
        return partial, elapsed

    def close(self) -> None:
        """Остановить процессы пула."""
        with self._lock:
            self._terminate_locked()


def _compute_chunk_star(args: tuple[dict, int, int]) -> tuple[int, int, float, list[float]]:
    return compute_chunk(*args)


def _warm_up_task(_i: int) -> int:
    """Пустая задача: заставляет процесс пула загрузить этот модуль (и NumPy)."""
    return int(np.ones(1)[0])
