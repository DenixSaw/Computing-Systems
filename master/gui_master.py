"""Окно Master: параметры задачи, таблица узлов, запуск расчёта, итоги, CSV и графики."""

from __future__ import annotations

import math
import socket
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QProgressBar,
                             QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
                             QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from common import config
from common.discovery import local_ipv4_addresses, primary_ip
from common.integrator import (ComputePool, TaskParams, exact_test_integral, partition_layers,
                               validate_params)
from common.qt_utils import CallThread, append_log
from master.live_plot import LiveProfilePlot
from master.plots import PlotResult, build_series_plots
from master.results_store import ResultsStore
from master.server import MASTER_NODE_ID, Assignment, MasterServer, RunWorker

# Колонки таблицы узлов
COL_NAME, COL_IP, COL_CORES, COL_SHARE, COL_STATUS, COL_PROGRESS, COL_TIME = range(7)
NODE_HEADERS = ["Имя", "IP", "Ядра", "Доля (слои)", "Статус", "Прогресс", "Время, с"]
SUMMARY_HEADERS = ["Узел", "IP", "Ядра", "Слои", "Доля, %", "Время узла, с", "Частичная сумма"]


class MasterWindow(QMainWindow):
    """Главное окно режима Master."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Master — распределённый тройной интеграл (Симпсон)")
        self.resize(1400, 900)

        self.hostname = socket.gethostname()
        self.master_ip = primary_ip()
        self.pool = ComputePool(config.cpu_count())
        self.server = MasterServer(self.hostname)
        self.server.log.connect(self.log)
        self.server.node_connected.connect(self._on_node_connected)
        self.server.node_disconnected.connect(self._on_node_disconnected)

        # узлы: node_id -> данные строки таблицы
        self.nodes: dict[int, dict[str, Any]] = {}
        self._gone: set[int] = set()            # отключились во время расчёта
        self.worker: RunWorker | None = None
        self.run_meta: dict[str, Any] = {}
        self._progress: dict[int, float] = {}
        self._plot_threads: list[CallThread] = []
        self._warm_thread: CallThread | None = None
        self._series_manual = False
        self.runs_dir = config.DEFAULT_RUNS_DIR

        self._build_ui()
        self._add_master_row()
        self._on_function_changed()
        self._update_runs_label()
        self._start_server()
        self._warm_up_pool()

    # =====================================================================
    # Интерфейс
    # =====================================================================

    def _build_ui(self) -> None:
        # ---- левая колонка: параметры ------------------------------------
        task_box = QGroupBox("Задача")
        form = QFormLayout(task_box)

        self.func_combo = QComboBox()
        for key, title in config.FUNCTIONS.items():
            self.func_combo.addItem(title, key)
        self.func_combo.currentIndexChanged.connect(self._on_function_changed)
        form.addRow("Функция:", self.func_combo)

        bounds = QGridLayout()
        self.bound_spins: list[QDoubleSpinBox] = []
        for row, axis in enumerate("xyz"):
            bounds.addWidget(QLabel(f"{axis}:"), row, 0)
            for col in range(2):
                sb = QDoubleSpinBox()
                sb.setDecimals(6)
                sb.setRange(-1e6, 1e6)
                sb.setSingleStep(0.1)
                bounds.addWidget(sb, row, 1 + 2 * col)
                if col == 0:
                    bounds.addWidget(QLabel("…"), row, 2)
                self.bound_spins.append(sb)
        form.addRow("Область:", bounds)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(2, config.N_MAX)
        self.n_spin.setSingleStep(2)
        self.n_spin.setValue(config.N_DEFAULT)
        self.n_spin.valueChanged.connect(self._update_auto_series)
        self.n_spin.editingFinished.connect(self._make_n_even)
        form.addRow("N (чётное):", self.n_spin)
        hint = QLabel(f"Быстрый тест: N = {config.N_QUICK}; демонстрация: N = {config.N_DEMO} "
                      f"(тяжёлая функция ≈ 1–3 мин на одном ядре). Время ~ N³.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        form.addRow(hint)

        self.split_combo = QComboBox()
        for key, title in config.SPLIT_MODES.items():
            self.split_combo.addItem(title, key)
        self.split_combo.currentIndexChanged.connect(self._update_auto_series)
        form.addRow("Деление:", self.split_combo)

        self.proc_spin = QSpinBox()
        self.proc_spin.setRange(1, max(256, config.cpu_count()))
        self.proc_spin.setValue(config.cpu_count())
        self.proc_spin.editingFinished.connect(self._on_processes_changed)
        form.addRow("Процессов master:", self.proc_spin)

        series_row = QHBoxLayout()
        self.series_edit = QLineEdit()
        self.series_edit.textEdited.connect(self._on_series_edited)
        series_row.addWidget(self.series_edit)
        reset = QToolButton()
        reset.setText("↺")
        reset.setToolTip("Вернуть автоматическое название")
        reset.clicked.connect(self._reset_series)
        series_row.addWidget(reset)
        form.addRow("Название серии:", series_row)

        net_box = QGroupBox("Сеть")
        net = QFormLayout(net_box)
        ips = ", ".join(local_ipv4_addresses()) or self.master_ip
        ip_label = QLabel(f"<b>{self.master_ip}</b>  (все адреса: {ips})")
        ip_label.setWordWrap(True)
        ip_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        net.addRow("Имя master:", QLabel(self.hostname))
        net.addRow("IP master:", ip_label)
        self.tcp_spin = QSpinBox()
        self.tcp_spin.setRange(1, 65535)
        self.tcp_spin.setValue(config.TCP_PORT)
        net.addRow("TCP-порт:", self.tcp_spin)
        self.udp_spin = QSpinBox()
        self.udp_spin.setRange(1, 65535)
        self.udp_spin.setValue(config.UDP_PORT)
        net.addRow("UDP-порт автопоиска:", self.udp_spin)
        self.group_edit = QLineEdit(config.DEFAULT_GROUP)
        net.addRow("Код группы:", self.group_edit)
        self.restart_btn = QPushButton("Применить и перезапустить сервер")
        self.restart_btn.clicked.connect(self._start_server)
        net.addRow(self.restart_btn)
        self.server_label = QLabel("—")
        net.addRow("Сервер:", self.server_label)

        res_box = QGroupBox("Результаты (CSV и графики)")
        res = QVBoxLayout(res_box)
        self.runs_label = QLabel()
        self.runs_label.setWordWrap(True)
        self.runs_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        res.addWidget(self.runs_label)
        row = QHBoxLayout()
        change = QPushButton("Изменить…")
        change.clicked.connect(self._choose_runs_dir)
        open_btn = QPushButton("Открыть папку с результатами")
        open_btn.clicked.connect(self._open_runs_dir)
        row.addWidget(change)
        row.addWidget(open_btn)
        res.addLayout(row)

        self.start_btn = QPushButton("▶ Запустить расчёт")
        self.stop_btn = QPushButton("■ Остановить")
        for b in (self.start_btn, self.stop_btn):
            b.setMinimumHeight(40)
            f = b.font()
            f.setPointSize(f.pointSize() + 1)
            f.setBold(True)
            b.setFont(f)
        self.start_btn.clicked.connect(self.start_run)
        self.stop_btn.clicked.connect(self.stop_run)
        self.stop_btn.setEnabled(False)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(task_box)
        lv.addWidget(net_box)
        lv.addWidget(res_box)
        btns = QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        lv.addLayout(btns)
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)

        # ---- правая колонка: узлы, результат, итоги, журнал -------------
        nodes_box = QGroupBox("Узлы")
        nv = QVBoxLayout(nodes_box)
        self.nodes_table = QTableWidget(0, len(NODE_HEADERS))
        self.nodes_table.setHorizontalHeaderLabels(NODE_HEADERS)
        self._setup_table(self.nodes_table)
        nv.addWidget(self.nodes_table)
        self.total_progress = QProgressBar()
        self.total_progress.setFormat("Общий прогресс: %p %")
        nv.addWidget(self.total_progress)

        result_box = QGroupBox("Результат")
        rf = QFormLayout(result_box)
        big = QFont()
        big.setPointSize(big.pointSize() + 3)
        big.setBold(True)
        self.integral_label = QLabel("—")
        self.integral_label.setFont(big)
        self.integral_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.time_label = QLabel("—")
        self.time_label.setFont(big)
        self.error_label = QLabel("—")
        self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rf.addRow("Интеграл:", self.integral_label)
        rf.addRow("Общее время (wall-clock):", self.time_label)
        rf.addRow("Погрешность:", self.error_label)

        summary_box = QWidget()
        sv = QVBoxLayout(summary_box)
        self.summary_label = QLabel("Расчёт ещё не выполнялся")
        self.summary_label.setWordWrap(True)
        sv.addWidget(self.summary_label)
        self.summary_table = QTableWidget(0, len(SUMMARY_HEADERS))
        self.summary_table.setHorizontalHeaderLabels(SUMMARY_HEADERS)
        self._setup_table(self.summary_table)
        sv.addWidget(self.summary_table)

        log_box = QGroupBox("Журнал событий")
        lgv = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        lgv.addWidget(self.log_view)

        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(nodes_box, 3)
        tv.addWidget(result_box, 0)

        # Вкладки: живой профиль расчёта и итоги последнего запуска
        self.live_plot = LiveProfilePlot()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.live_plot, "Профиль расчёта (live)")
        self.tabs.addTab(summary_box, "Итоги последнего запуска по узлам")

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(top)
        right.addWidget(self.tabs)
        right.addWidget(log_box)
        right.setSizes([330, 380, 190])

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(scroll)
        main.addWidget(right)
        main.setSizes([450, 950])
        self.setCentralWidget(main)

    @staticmethod
    def _setup_table(table: QTableWidget) -> None:
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

    def log(self, text: str) -> None:
        append_log(self.log_view, text)

    # =====================================================================
    # Параметры
    # =====================================================================

    def _function(self) -> str:
        return str(self.func_combo.currentData())

    def _split_mode(self) -> str:
        return str(self.split_combo.currentData())

    def _on_function_changed(self) -> None:
        for sb, value in zip(self.bound_spins, config.DEFAULT_BOUNDS[self._function()]):
            sb.setValue(value)
        self._update_auto_series()

    def _make_n_even(self) -> None:
        if self.n_spin.value() % 2:
            self.n_spin.setValue(self.n_spin.value() + 1)

    def _auto_series(self) -> str:
        n = self.n_spin.value() + (self.n_spin.value() % 2)
        return f"{self._function()}_N{n}_{self._split_mode()}"

    def _update_auto_series(self) -> None:
        if not self._series_manual:
            self.series_edit.setText(self._auto_series())

    def _on_series_edited(self, _text: str) -> None:
        self._series_manual = True

    def _reset_series(self) -> None:
        self._series_manual = False
        self._update_auto_series()

    def _params(self) -> TaskParams:
        self._make_n_even()
        b = [sb.value() for sb in self.bound_spins]
        return TaskParams(self._function(), self.n_spin.value(), *b)

    def _set_params_enabled(self, enabled: bool) -> None:
        for w in (self.func_combo, self.n_spin, self.split_combo, self.proc_spin,
                  self.series_edit, self.restart_btn, *self.bound_spins):
            w.setEnabled(enabled)

    # --- пул процессов master ----------------------------------------------

    def _on_processes_changed(self) -> None:
        if self.proc_spin.value() != self.pool.processes:
            self.pool.set_processes(self.proc_spin.value())
            node = self.nodes.get(MASTER_NODE_ID)
            if node:
                node["cores"] = self.proc_spin.value()
                self._refresh_row(MASTER_NODE_ID)
            self._warm_up_pool()

    def _warm_up_pool(self) -> None:
        """Заранее запустить процессы пула, чтобы их старт не попал в замер времени."""
        if self._warm_thread is not None and self._warm_thread.isRunning():
            return
        n = self.pool.processes
        t0 = time.perf_counter()
        self._warm_thread = CallThread(self.pool.warm_up)
        self._warm_thread.done.connect(
            lambda _r: self.log(f"Пул master готов: {n} процессов "
                                f"(запуск занял {time.perf_counter() - t0:.1f} с)"))
        self._warm_thread.failed.connect(lambda e: self.log(f"⚠ Ошибка запуска пула: {e}"))
        self._warm_thread.start()

    # --- папка результатов -----------------------------------------------------

    def _update_runs_label(self) -> None:
        self.runs_label.setText(f"Папка: {self.runs_dir}")

    def _choose_runs_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка для результатов", str(self.runs_dir))
        if path:
            self.runs_dir = Path(path)
            self._update_runs_label()
            self.log(f"Папка результатов: {self.runs_dir}")

    def _open_runs_dir(self) -> None:
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.log(f"✖ Не удалось создать папку {self.runs_dir}: {exc}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.runs_dir)))

    # =====================================================================
    # Сервер и узлы
    # =====================================================================

    def _start_server(self) -> None:
        if self.worker is not None:
            return
        if self.server.slaves_count():
            self.log("⚠ Перезапуск сервера отключит подключённые slave — им нужно подключиться заново")
        group = self.group_edit.text().strip() or config.DEFAULT_GROUP
        self.group_edit.setText(group)
        ok = self.server.start(self.tcp_spin.value(), self.udp_spin.value(), group)
        if ok:
            self.server_label.setText(f"работает: TCP {self.tcp_spin.value()}, "
                                      f"UDP {self.udp_spin.value()}, группа «{group}»")
        else:
            self.server_label.setText("не запущен (порт занят) — только расчёт на master")

    def _add_master_row(self) -> None:
        self.nodes[MASTER_NODE_ID] = {
            "name": f"{self.hostname} (master)", "ip": self.master_ip,
            "cores": self.proc_spin.value(), "share": "—", "status": "готов",
            "progress": 0.0, "time": None, "is_master": True,
        }
        self._rebuild_table()

    def _on_node_connected(self, node_id: int, name: str, ip: str, cores: int) -> None:
        self.nodes[node_id] = {"name": name, "ip": ip, "cores": cores, "share": "—",
                               "status": "подключён", "progress": 0.0, "time": None,
                               "is_master": False}
        self._rebuild_table()
        self.log(f"Узлов в системе: {len(self.nodes)} (master + {len(self.nodes) - 1} slave)")

    def _on_node_disconnected(self, node_id: int, _name: str, _reason: str) -> None:
        if node_id not in self.nodes:
            return
        in_run = self.worker is not None and any(
            a.node_id == node_id for a in self.run_meta.get("assignments", []))
        if in_run:
            self._gone.add(node_id)
            self.nodes[node_id]["status"] = "отключён"
            self._refresh_row(node_id)
        else:
            del self.nodes[node_id]
            self._rebuild_table()

    def _rebuild_table(self) -> None:
        ids = sorted(self.nodes)
        self.nodes_table.setRowCount(len(ids))
        self._row_of: dict[int, int] = {}
        for row, node_id in enumerate(ids):
            self._row_of[node_id] = row
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(True)
            self.nodes_table.setCellWidget(row, COL_PROGRESS, bar)
            self._refresh_row(node_id)

    def _refresh_row(self, node_id: int) -> None:
        row = getattr(self, "_row_of", {}).get(node_id)
        node = self.nodes.get(node_id)
        if row is None or node is None:
            return
        values = {
            COL_NAME: node["name"], COL_IP: node["ip"], COL_CORES: str(node["cores"]),
            COL_SHARE: node["share"], COL_STATUS: node["status"],
            COL_TIME: "" if node["time"] is None else f"{node['time']:.3f}",
        }
        for col, text in values.items():
            item = QTableWidgetItem(text)
            if node["is_master"]:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.nodes_table.setItem(row, col, item)
        bar = self.nodes_table.cellWidget(row, COL_PROGRESS)
        if isinstance(bar, QProgressBar):
            bar.setValue(int(node["progress"] * 1000))
            bar.setFormat(f"{node['progress'] * 100:.0f} %")

    # =====================================================================
    # Расчёт
    # =====================================================================

    def start_run(self) -> None:
        """Нажата «Запустить расчёт»: статическое деление и старт RunWorker."""
        t_click = time.perf_counter()
        if self.worker is not None:
            return
        params = self._params()
        err = validate_params(params)
        if err:
            self.log(f"✖ {err}")
            return
        self._on_processes_changed()
        series = self.series_edit.text().strip() or self._auto_series()
        self.series_edit.setText(series)
        split_mode = self._split_mode()

        # Узлы: master (узел 0) + все зарегистрированные slave
        self.nodes[MASTER_NODE_ID]["cores"] = self.pool.processes
        candidates = [(MASTER_NODE_ID, self.nodes[MASTER_NODE_ID]["name"], self.master_ip,
                       self.pool.processes, True)]
        for c in sorted(self.server.slaves(), key=lambda c: c.node_id):
            if c.node_id in self.nodes:
                candidates.append((c.node_id, c.name, c.ip, c.cores, False))
        weights = [cores if split_mode == "cores" else 1 for *_x, cores, _m in candidates]
        ranges = partition_layers(params.n, weights)

        assignments: list[Assignment] = []
        for (node_id, name, ip, cores, is_master), (s, e) in zip(candidates, ranges):
            node = self.nodes[node_id]
            node.update(progress=0.0, time=None)
            if e <= s:
                node.update(share="—", status="не участвует")
                self.log(f"⚠ Узлу «{name}» не досталось слоёв (N слишком мало для числа узлов)")
                continue
            node.update(share=f"{s}–{e - 1} ({(e - s) / (params.n + 1) * 100:.1f} %)",
                        status="ожидает")
            assignments.append(Assignment(node_id, name, ip, cores, is_master, s, e))
        self._rebuild_table()

        self.run_meta = {"params": params, "series": series, "split_mode": split_mode,
                         "assignments": assignments}
        self._progress = {a.node_id: 0.0 for a in assignments}
        self._gone.clear()
        self.total_progress.setValue(0)
        self.integral_label.setText("считается…")
        self.time_label.setText("—")
        self.error_label.setText("—")

        mode_text = config.SPLIT_MODES[split_mode].lower()
        self.log(f"=== Запуск: функция {params.function}, N={params.n}, деление {mode_text}, "
                 f"серия «{series}», узлов: {len(assignments)} ===")
        for a in assignments:
            self.log(f"  {a.name}: слои {a.start}–{a.end - 1} ({a.layers} шт.), ядер {a.cores}")

        self.worker = RunWorker(self.server, self.pool, params, assignments, t_click)
        self.worker.log.connect(self.log)
        self.worker.node_status.connect(self._on_node_status)
        self.worker.node_progress.connect(self._on_node_progress)
        self.worker.node_done.connect(self._on_node_done)
        self.worker.node_chunks.connect(self.live_plot.add_chunks)
        self.worker.run_finished.connect(self._on_run_finished)
        self.live_plot.reset(params, assignments)
        self.tabs.setCurrentWidget(self.live_plot)
        self._set_params_enabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def stop_run(self) -> None:
        if self.worker is not None:
            self.log("Остановка расчёта…")
            self.worker.cancel()
            self.stop_btn.setEnabled(False)

    def _on_node_status(self, node_id: int, status: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id]["status"] = status
            self._refresh_row(node_id)

    def _on_node_progress(self, node_id: int, frac: float) -> None:
        if node_id in self.nodes:
            self.nodes[node_id]["progress"] = frac
            self._refresh_row(node_id)
        self._progress[node_id] = frac
        params: TaskParams = self.run_meta["params"]
        done = sum(self._progress.get(a.node_id, 0.0) * a.layers
                   for a in self.run_meta["assignments"])
        self.total_progress.setValue(int(100 * done / (params.n + 1)))

    def _on_node_done(self, node_id: int, _partial: float, node_time: float) -> None:
        if node_id in self.nodes:
            self.nodes[node_id]["time"] = node_time
            self._refresh_row(node_id)

    def _on_run_finished(self, result: dict[str, Any]) -> None:
        """Расчёт завершён: показать итоги, записать CSV, построить графики."""
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.wait(2000)
        self._set_params_enabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        params: TaskParams = self.run_meta["params"]
        assignments: list[Assignment] = result["assignments"]
        status: str = result["status"]
        total_time: float = result["total_time"]
        integral: float | None = result["integral"]

        abs_err = rel_err = None
        if integral is not None:
            self.total_progress.setValue(100)
            self.integral_label.setText(f"{integral:.15g}")
            if params.function == "test":
                exact = exact_test_integral(params.ax, params.bx, params.ay, params.by,
                                            params.az, params.bz)
                abs_err = abs(integral - exact)
                rel_err = abs_err / abs(exact) if exact != 0 else None
                rel_text = f"{rel_err:.3e}" if rel_err is not None else "—"
                self.error_label.setText(f"точное {exact:.15g}; абсолютная {abs_err:.3e}; "
                                         f"относительная {rel_text}")
            else:
                self.error_label.setText("— (для тяжёлой функции точное значение неизвестно)")
        else:
            self.integral_label.setText({"stopped": "остановлено", "failed": "ошибка"}
                                        .get(status, status))
        self.time_label.setText(f"{total_time:.3f} с")

        status_text = {"ok": "успешно", "recovered": "с пересчётом доли выбывшего узла",
                       "stopped": "остановлен пользователем", "failed": "ошибка"}[status]
        self.log(f"=== Расчёт завершён ({status_text}) за {total_time:.3f} с"
                 + (f", интеграл = {integral:.15g}" if integral is not None else "") + " ===")
        if result.get("error"):
            self.log(f"✖ {result['error']}")
        if status == "recovered":
            self.log("⚠ Запуск записан со статусом recovered и не попадёт на графики серии")

        self._fill_summary(params, assignments, total_time, status)
        self.live_plot.finish(integral, status)

        # Удалить из таблицы узлы, отключившиеся во время расчёта
        for node_id in self._gone:
            self.nodes.pop(node_id, None)
        if self._gone:
            self._rebuild_table()
        self._gone.clear()

        self._save_and_plot(params, assignments, status, total_time, integral, abs_err, rel_err)

    def _fill_summary(self, params: TaskParams, assignments: list[Assignment],
                      total_time: float, status: str) -> None:
        total_cores = sum(a.cores for a in assignments)
        self.summary_label.setText(
            f"<b>Узлов: {len(assignments)}</b> (ядер всего: {total_cores}) · функция: "
            f"{params.function} · N = {params.n} · деление: "
            f"{config.SPLIT_MODES[self.run_meta['split_mode']].lower()} · серия: "
            f"«{self.run_meta['series']}» · общее время: <b>{total_time:.3f} с</b> · статус: {status}")
        self.summary_table.setRowCount(len(assignments))
        for row, a in enumerate(assignments):
            name = a.name + (" (пересчитано master)" if a.recovered else "")
            values = [name, a.ip, str(a.cores), f"{a.start}–{a.end - 1}",
                      f"{a.layers / (params.n + 1) * 100:.2f}",
                      "" if a.node_time is None else f"{a.node_time:.3f}",
                      "" if a.partial is None else f"{a.partial:.12g}"]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if a.is_master:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self.summary_table.setItem(row, col, item)

    def _save_and_plot(self, params: TaskParams, assignments: list[Assignment], status: str,
                       total_time: float, integral: float | None,
                       abs_err: float | None, rel_err: float | None) -> None:
        """Сначала CSV (сразу, с flush), затем графики в отдельном потоке."""
        series = self.run_meta["series"]
        store = ResultsStore(self.runs_dir, log=self.log)
        run_row = {
            "function": params.function, "N": params.n, "split_mode": self.run_meta["split_mode"],
            "series": series, "nodes_count": len(assignments),
            "nodes_list": ";".join(a.name for a in assignments),
            "total_cores": sum(a.cores for a in assignments),
            "total_time_s": total_time, "integral_value": integral,
            "abs_error": abs_err, "rel_error": rel_err, "status": status,
        }
        node_rows = [{
            "node_name": a.name, "ip": a.ip, "cores": a.cores,
            "slice_start": a.start, "slice_end": a.end - 1,
            "share_percent": round(a.layers / (params.n + 1) * 100, 4),
            "node_time_s": a.node_time, "partial_sum": a.partial, "is_master": a.is_master,
        } for a in assignments]
        saved = store.save_run(run_row, node_rows)
        if not saved.run_id:
            return
        where = ", ".join(str(p) for p in (saved.runs_path, saved.nodes_path) if p)
        self.log(f"Запуск №{saved.run_id} сохранён: {where}")

        run_id = saved.run_id
        runs_dir = self.runs_dir
        thread = CallThread(lambda: build_series_plots(runs_dir, series, run_id))
        thread.done.connect(self._on_plots_done)
        thread.failed.connect(lambda e: self.log(f"⚠ Ошибка построения графиков: {e}"))
        thread.finished.connect(lambda t=thread: self._plot_threads.remove(t)
                                if t in self._plot_threads else None)
        self._plot_threads.append(thread)
        thread.start()

    def _on_plots_done(self, result: PlotResult) -> None:
        for msg in result.messages:
            self.log(msg)
        if result.files:
            self.log("Графики и сводка сохранены:")
            for path in result.files:
                self.log(f"  {path}")

    # =====================================================================
    # Завершение
    # =====================================================================

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — имя метода Qt
        """Корректное завершение: остановить расчёт, закрыть сокеты, процессы и потоки."""
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(15000)
        self.server.stop(notify_slaves=True)
        for t in list(self._plot_threads):
            t.wait(30000)
        if self._warm_thread is not None:
            self._warm_thread.wait(30000)
        self.pool.close()
        event.accept()
