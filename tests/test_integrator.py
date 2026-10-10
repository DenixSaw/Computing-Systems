"""Проверка интегратора: распределённый расчёт совпадает с последовательным."""

from __future__ import annotations

import math
import threading
import unittest

import numpy as np

from common.integrator import (ComputeCancelled, ComputePool, TaskParams, compute_range,
                               exact_test_integral, integrate_sequential, partition_layers,
                               simpson_weights, validate_params)

TEST_BOUNDS = (0.0, math.pi, 0.0, math.pi / 2, 0.0, 1.0)


def naive_simpson(p: TaskParams) -> float:
    """Эталон «в лоб»: вся сетка N³ сразу (только для малых N)."""
    f = {"test": lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(z),
         "heavy": lambda x, y, z: (np.sin(x * y) * np.exp(-z)
                                   + np.sqrt(x * x + y * y + z * z + 1) * np.cos(x + y * z)
                                   + np.log(1 + x * y * z))}[p.function]
    x = np.linspace(p.ax, p.bx, p.n + 1)
    y = np.linspace(p.ay, p.by, p.n + 1)
    z = np.linspace(p.az, p.bz, p.n + 1)
    w = simpson_weights(p.n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    ww = w[:, None, None] * w[None, :, None] * w[None, None, :]
    return float(np.sum(ww * f(xx, yy, zz))) * p.scale


class TestWeightsAndPartition(unittest.TestCase):
    def test_weights(self) -> None:
        self.assertEqual(simpson_weights(6).tolist(), [1, 4, 2, 4, 2, 4, 1])

    def test_partition_covers_all_layers(self) -> None:
        for n in (2, 4, 10, 50, 100, 101 * 2):
            for weights in ([1], [1, 1], [8, 4, 4, 2], [1, 1, 1, 1, 1], [16, 1], [3, 0, 3]):
                ranges = partition_layers(n, weights)
                self.assertEqual(len(ranges), len(weights))
                pos = 0
                for s, e in ranges:
                    self.assertEqual(s, pos, "диапазоны должны идти подряд без дыр")
                    self.assertGreaterEqual(e, s)
                    if e > s:
                        self.assertEqual(s % 2, 0, "граница должна быть чётной (панель Симпсона)")
                    pos = e
                self.assertEqual(pos, n + 1, "должны быть покрыты все слои 0..N")

    def test_partition_proportional(self) -> None:
        ranges = partition_layers(400, [8, 4, 4, 0])
        sizes = [e - s for s, e in ranges]
        self.assertEqual(sizes[3], 0)
        self.assertAlmostEqual(sizes[0] / sum(sizes), 0.5, delta=0.01)
        self.assertAlmostEqual(sizes[1], sizes[2], delta=2)

    def test_more_nodes_than_panels(self) -> None:
        ranges = partition_layers(4, [1, 1, 1, 1])
        self.assertEqual(sum(1 for s, e in ranges if e > s), 2)

    def test_validate(self) -> None:
        self.assertIsNone(validate_params(TaskParams("test", 10, *TEST_BOUNDS)))
        self.assertIsNotNone(validate_params(TaskParams("test", 11, *TEST_BOUNDS)))
        self.assertIsNotNone(validate_params(TaskParams("heavy", 10, -1, 1, 0, 1, 0, 1)))
        self.assertIsNotNone(validate_params(TaskParams("test", 10, 1, 0, 0, 1, 0, 1)))


class TestIntegration(unittest.TestCase):
    def test_matches_naive(self) -> None:
        for func, bounds in (("test", TEST_BOUNDS), ("heavy", (0.0, 2.0, 0.0, 2.0, 0.0, 2.0))):
            p = TaskParams(func, 20, *bounds)
            self.assertAlmostEqual(integrate_sequential(p), naive_simpson(p), places=10)

    def test_accuracy_test_function(self) -> None:
        exact = exact_test_integral(*TEST_BOUNDS)
        self.assertAlmostEqual(exact, 2 * (math.e - 1), places=12)
        value = integrate_sequential(TaskParams("test", 40, *TEST_BOUNDS))
        self.assertLess(abs(value - exact) / exact, 1e-6)

    def test_distributed_equals_sequential(self) -> None:
        """Сумма частичных результатов узлов = последовательный расчёт (малое N)."""
        for func, bounds in (("test", TEST_BOUNDS), ("heavy", (0.0, 2.0, 0.0, 2.0, 0.0, 2.0))):
            for n in (2, 6, 30, 64):
                p = TaskParams(func, n, *bounds)
                seq = integrate_sequential(p)
                for weights in ([1], [1, 1], [4, 2, 2, 1], [1, 1, 1, 1, 1], [12, 8, 4, 4]):
                    parts = [compute_range(p, s, e) for s, e in partition_layers(n, weights)]
                    self.assertAlmostEqual(math.fsum(parts), seq, delta=1e-12 * max(1, abs(seq)),
                                           msg=f"{func} N={n} weights={weights}")


class TestComputePool(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pool = ComputePool(2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.pool.close()

    def test_pool_equals_sequential(self) -> None:
        p = TaskParams("heavy", 40, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
        progress: list[float] = []
        total = 0.0
        for s, e in partition_layers(p.n, [2, 1]):
            partial, elapsed = self.pool.compute(p, s, e, progress_cb=progress.append)
            self.assertAlmostEqual(partial, compute_range(p, s, e), delta=1e-12)
            self.assertGreaterEqual(elapsed, 0.0)
            total += partial
        self.assertAlmostEqual(total, integrate_sequential(p), delta=1e-11)
        self.assertAlmostEqual(progress[-1], 1.0)

    def test_cancel(self) -> None:
        p = TaskParams("heavy", 40, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(ComputeCancelled):
            self.pool.compute(p, 0, p.n + 1, cancel_event=cancel)
        # после отмены пул пересоздаётся и снова работает
        partial, _ = self.pool.compute(p, 0, p.n + 1)
        self.assertAlmostEqual(partial, integrate_sequential(p), delta=1e-11)


if __name__ == "__main__":
    unittest.main()
