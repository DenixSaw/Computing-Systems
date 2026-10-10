"""Проверка записи CSV и построения графиков по ним."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from master.plots import build_series_plots, safe_series_name, series_dir
from master.results_store import NODES_COLUMNS, RUNS_COLUMNS, ResultsStore


def make_run(series: str, nodes_count: int, total_time: float, status: str = "ok",
             function: str = "heavy", n: int = 200) -> tuple[dict, list[dict]]:
    run = {"function": function, "N": n, "split_mode": "cores", "series": series,
           "nodes_count": nodes_count,
           "nodes_list": ";".join(f"node{i}" for i in range(nodes_count)),
           "total_cores": 4 * nodes_count, "total_time_s": total_time,
           "integral_value": 12.5, "abs_error": None, "rel_error": None, "status": status}
    nodes = [{"node_name": f"node{i}", "ip": f"10.0.0.{i + 1}", "cores": 4,
              "slice_start": i * 10, "slice_end": i * 10 + 9,
              "share_percent": 100 / nodes_count, "node_time_s": total_time * 0.9,
              "partial_sum": 12.5 / nodes_count, "is_master": i == 0}
             for i in range(nodes_count)]
    return run, nodes


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestResultsStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "runs"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_header_once_and_append(self) -> None:
        store = ResultsStore(self.dir)
        for k in range(3):
            res = store.save_run(*make_run("s", 1 + k, 10.0 / (k + 1)))
            self.assertEqual(res.run_id, k + 1)
            self.assertFalse(res.used_fallback)
        lines = store.runs_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(RUNS_COLUMNS))
        self.assertEqual(len(lines), 4)
        self.assertEqual(store.nodes_path.read_text(encoding="utf-8").splitlines()[0],
                         ",".join(NODES_COLUMNS))
        rows = read_rows(store.runs_path)
        self.assertEqual([r["run_id"] for r in rows], ["1", "2", "3"])
        self.assertEqual(rows[0]["abs_error"], "")
        self.assertEqual(len(read_rows(store.nodes_path)), 1 + 2 + 3)
        self.assertEqual(read_rows(store.nodes_path)[0]["is_master"], "true")

    def test_run_id_continues_after_restart(self) -> None:
        ResultsStore(self.dir).save_run(*make_run("s", 1, 5.0))
        ResultsStore(self.dir).save_run(*make_run("s", 2, 3.0))
        res = ResultsStore(self.dir).save_run(*make_run("s", 3, 2.0))   # «новый запуск программы»
        self.assertEqual(res.run_id, 3)

    def test_fallback_when_locked(self) -> None:
        store = ResultsStore(self.dir)
        store.save_run(*make_run("s", 1, 5.0))
        original = ResultsStore._append

        def fake_append(path: Path, columns, rows):
            if path.name in ("runs.csv", "run_nodes.csv"):
                raise PermissionError(13, "Permission denied (файл открыт в Excel)")
            return original(path, columns, rows)

        messages: list[str] = []
        store = ResultsStore(self.dir, log=messages.append)
        with mock.patch.object(ResultsStore, "_append", staticmethod(fake_append)):
            res = store.save_run(*make_run("s", 2, 3.0))
        self.assertTrue(res.used_fallback)
        self.assertTrue(res.runs_path.name.startswith("runs_"))
        self.assertTrue(res.runs_path.exists())
        self.assertTrue(messages)
        # нумерация учитывает запасной файл
        self.assertEqual(ResultsStore(self.dir).next_run_id(), 3)

    def test_safe_series_name(self) -> None:
        self.assertEqual(safe_series_name('a/b:c*?"<>|'), "a_b_c______")
        self.assertEqual(safe_series_name("  "), "unnamed")


class TestPlots(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "runs"
        self.store = ResultsStore(self.dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_full_series(self) -> None:
        self.store.save_run(*make_run("demo", 1, 100.0))
        self.store.save_run(*make_run("demo", 2, 60.0))
        self.store.save_run(*make_run("demo", 2, 52.0))          # повтор заменяет предыдущий
        self.store.save_run(*make_run("demo", 3, 999.0, status="stopped"))
        self.store.save_run(*make_run("other", 4, 1.0))          # другая серия
        res = self.store.save_run(*make_run("demo", 4, 28.0))
        result = build_series_plots(self.dir, "demo", res.run_id)
        out = series_dir(self.dir, "demo")
        for name in ("time_vs_nodes.png", "speedup_vs_nodes.png", "efficiency_vs_nodes.png",
                     "summary.csv", f"run_{res.run_id}/nodes_time.png"):
            self.assertTrue((out / name).exists(), name)
        summary = read_rows(out / "summary.csv")
        self.assertEqual([r["nodes_count"] for r in summary], ["1", "2", "4"])
        self.assertEqual(summary[1]["run_id"], "3")
        self.assertAlmostEqual(float(summary[2]["speedup"]), 100 / 28)
        self.assertAlmostEqual(float(summary[2]["efficiency"]), 100 / 28 / 4)
        self.assertFalse(any("некорректны" in m for m in result.messages))

    def test_without_single_node_run(self) -> None:
        res = self.store.save_run(*make_run("nosingle", 2, 10.0))
        result = build_series_plots(self.dir, "nosingle", res.run_id)
        out = series_dir(self.dir, "nosingle")
        self.assertTrue((out / "time_vs_nodes.png").exists())
        self.assertFalse((out / "speedup_vs_nodes.png").exists())
        self.assertTrue(any("только на master" in m for m in result.messages))

    def test_mixed_series_warning(self) -> None:
        self.store.save_run(*make_run("mix", 1, 10.0, n=200))
        res = self.store.save_run(*make_run("mix", 2, 6.0, n=400))
        result = build_series_plots(self.dir, "mix", res.run_id)
        self.assertTrue(any("некорректны" in m for m in result.messages))

    def test_no_ok_runs(self) -> None:
        res = self.store.save_run(*make_run("bad", 1, 1.0, status="failed"))
        result = build_series_plots(self.dir, "bad", res.run_id)
        self.assertTrue(result.messages)   # не падает, просто сообщает


if __name__ == "__main__":
    unittest.main()
