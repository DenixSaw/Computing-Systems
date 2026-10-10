"""Окно Slave: автопоиск master, подключение, текущая задача, прогресс, журнал."""

from __future__ import annotations

import socket
import time
import uuid

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QFormLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
                             QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from common import config
from common.discovery import discover_masters, primary_ip
from common.integrator import ComputePool
from common.qt_utils import CallThread, append_log
from slave.client import SlaveClient

NOT_FOUND_HINT = ("Master не найден: проверьте, что компьютеры в одной сети и разрешён "
                  "UDP-порт {port}, либо введите IP вручную.")


class SlaveWindow(QMainWindow):
    """Главное окно режима Slave."""

    log_signal = pyqtSignal(str)   # журнал из рабочих потоков (потокобезопасно)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Slave — распределённый тройной интеграл (Симпсон)")
        self.resize(820, 860)
        self.pool = ComputePool(config.cpu_count())
        self.client: SlaveClient | None = None
        self.discovery_thread: CallThread | None = None
        self._warm_thread: CallThread | None = None
        self.found: list[dict] = []
        self.log_signal.connect(self.log)

        self._build_ui()
        self._set_connected(False)
        self.log(f"Slave запущен на {socket.gethostname()} ({primary_ip()}), "
                 f"ядер: {config.cpu_count()}")
        self._warm_up_pool()
        QTimer.singleShot(300, self.discover)   # автоматический поиск при открытии окна

    # =====================================================================
    # Интерфейс
    # =====================================================================

    def _build_ui(self) -> None:
        node_box = QGroupBox("Этот узел")
        nf = QFormLayout(node_box)
        self.name_edit = QLineEdit(f"{socket.gethostname()}-{uuid.uuid4().hex[:4]}")
        nf.addRow("Имя узла:", self.name_edit)
        self.proc_spin = QSpinBox()
        self.proc_spin.setRange(1, max(256, config.cpu_count()))
        self.proc_spin.setValue(config.cpu_count())
        self.proc_spin.editingFinished.connect(self._on_processes_changed)
        nf.addRow("Процессов для расчёта:", self.proc_spin)
        hint = QLabel("Это число сообщается master как «ядра» узла. Если на одном ПК запущено "
                      "несколько экземпляров, разделите между ними ядра.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        nf.addRow(hint)

        disc_box = QGroupBox("Автопоиск master в локальной сети (UDP)")
        dv = QVBoxLayout(disc_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Код группы:"))
        self.group_edit = QLineEdit(config.DEFAULT_GROUP)
        self.group_edit.setMaximumWidth(140)
        row.addWidget(self.group_edit)
        row.addWidget(QLabel("UDP-порт:"))
        self.udp_spin = QSpinBox()
        self.udp_spin.setRange(1, 65535)
        self.udp_spin.setValue(config.UDP_PORT)
        row.addWidget(self.udp_spin)
        self.find_btn = QPushButton("🔍 Найти master")
        self.find_btn.clicked.connect(self.discover)
        row.addWidget(self.find_btn)
        row.addStretch(1)
        dv.addLayout(row)

        self.found_table = QTableWidget(0, 4)
        self.found_table.setHorizontalHeaderLabels(["Имя master", "IP", "Порт", "Узлов подключено"])
        self.found_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.found_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.found_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.found_table.verticalHeader().setVisible(False)
        self.found_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.found_table.setMaximumHeight(150)
        self.found_table.cellDoubleClicked.connect(lambda r, _c: self._connect_found(r))
        dv.addWidget(self.found_table)

        self.auto_check = QCheckBox("Подключаться автоматически к единственному найденному master")
        self.auto_check.setChecked(True)
        dv.addWidget(self.auto_check)
        self.connect_found_btn = QPushButton("Подключиться к выбранному")
        self.connect_found_btn.clicked.connect(lambda: self._connect_found(self.found_table.currentRow()))
        dv.addWidget(self.connect_found_btn)
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #b35900; font-weight: bold;")
        self.hint_label.hide()
        dv.addWidget(self.hint_label)

        manual_box = QGroupBox("Ручное подключение (запасной вариант)")
        mv = QHBoxLayout(manual_box)
        mv.addWidget(QLabel("IP master:"))
        self.ip_edit = QLineEdit("127.0.0.1")
        mv.addWidget(self.ip_edit)
        mv.addWidget(QLabel("Порт:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(config.TCP_PORT)
        mv.addWidget(self.port_spin)
        self.connect_btn = QPushButton("Подключиться")
        self.connect_btn.clicked.connect(
            lambda: self.connect_to(self.ip_edit.text().strip(), self.port_spin.value()))
        mv.addWidget(self.connect_btn)
        self.disconnect_btn = QPushButton("Отключиться")
        self.disconnect_btn.clicked.connect(self.disconnect_from_master)
        mv.addWidget(self.disconnect_btn)

        state_box = QGroupBox("Состояние")
        sf = QFormLayout(state_box)
        self.status_label = QLabel("не подключён")
        f = self.status_label.font()
        f.setBold(True)
        self.status_label.setFont(f)
        sf.addRow("Статус:", self.status_label)
        self.task_label = QLabel("—")
        self.task_label.setWordWrap(True)
        sf.addRow("Текущая задача:", self.task_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setFormat("0 %")
        sf.addRow("Прогресс:", self.progress_bar)
        self.result_label = QLabel("—")
        sf.addRow("Последний результат:", self.result_label)

        log_box = QGroupBox("Журнал")
        lv = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        lv.addWidget(self.log_view)

        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(node_box)
        v.addWidget(disc_box)
        v.addWidget(manual_box)
        v.addWidget(state_box)
        v.addWidget(log_box, 1)
        self.setCentralWidget(central)

    def log(self, text: str) -> None:
        append_log(self.log_view, text)

    def _set_connected(self, connected: bool) -> None:
        busy = connected or self.client is not None
        for w in (self.connect_btn, self.connect_found_btn, self.find_btn, self.name_edit,
                  self.proc_spin, self.group_edit, self.udp_spin, self.ip_edit, self.port_spin):
            w.setEnabled(not busy)
        self.disconnect_btn.setEnabled(busy)

    # =====================================================================
    # Пул процессов
    # =====================================================================

    def _on_processes_changed(self) -> None:
        if self.proc_spin.value() != self.pool.processes:
            self.pool.set_processes(self.proc_spin.value())
            self._warm_up_pool()

    def _warm_up_pool(self) -> None:
        if self._warm_thread is not None and self._warm_thread.isRunning():
            return
        n = self.pool.processes
        t0 = time.perf_counter()
        self._warm_thread = CallThread(self.pool.warm_up)
        self._warm_thread.done.connect(
            lambda _r: self.log(f"Пул процессов готов: {n} "
                                f"(запуск занял {time.perf_counter() - t0:.1f} с)"))
        self._warm_thread.failed.connect(lambda e: self.log(f"⚠ Ошибка запуска пула: {e}"))
        self._warm_thread.start()

    # =====================================================================
    # Автопоиск
    # =====================================================================

    def discover(self) -> None:
        """Запустить UDP-поиск master (в отдельном потоке)."""
        if self.client is not None:
            return
        if self.discovery_thread is not None and self.discovery_thread.isRunning():
            return
        group = self.group_edit.text().strip() or config.DEFAULT_GROUP
        self.group_edit.setText(group)
        port = self.udp_spin.value()
        self.find_btn.setEnabled(False)
        self.find_btn.setText("Поиск…")
        self.hint_label.hide()
        self.found_table.setRowCount(0)
        self.discovery_thread = CallThread(
            lambda: discover_masters(group, port, log=self.log_signal.emit))
        self.discovery_thread.done.connect(self._on_discovered)
        self.discovery_thread.failed.connect(self._on_discovery_failed)
        self.discovery_thread.start()

    def _on_discovery_failed(self, error: str) -> None:
        self.find_btn.setText("🔍 Найти master")
        self._set_connected(False)
        self.log(f"✖ Ошибка автопоиска: {error}")
        self.hint_label.setText(NOT_FOUND_HINT.format(port=self.udp_spin.value()))
        self.hint_label.show()

    def _on_discovered(self, masters: list[dict]) -> None:
        self.find_btn.setText("🔍 Найти master")
        self._set_connected(self.client is not None)
        self.found = masters
        self.found_table.setRowCount(len(masters))
        for row, m in enumerate(masters):
            for col, value in enumerate((m["name"], m["ip"], m["tcp_port"], m["slaves"])):
                self.found_table.setItem(row, col, QTableWidgetItem(str(value)))
        if not masters:
            self.hint_label.setText(NOT_FOUND_HINT.format(port=self.udp_spin.value()))
            self.hint_label.show()
            return
        self.found_table.selectRow(0)
        first = masters[0]
        self.ip_edit.setText(first["ip"])
        self.port_spin.setValue(int(first["tcp_port"]))
        if len(masters) == 1:
            if self.auto_check.isChecked() and self.client is None:
                self.log("Найден один master — подключаюсь автоматически")
                self._connect_found(0)
            else:
                self.log("Найден один master — нажмите «Подключиться к выбранному»")
        else:
            self.log(f"Найдено несколько master ({len(masters)}) — выберите нужный в списке")

    def _connect_found(self, row: int) -> None:
        if 0 <= row < len(self.found):
            m = self.found[row]
            self.connect_to(m["ip"], int(m["tcp_port"]))
        else:
            self.log("Выберите master в списке найденных")

    # =====================================================================
    # Подключение
    # =====================================================================

    def connect_to(self, host: str, port: int) -> None:
        """Подключиться к master по TCP (в отдельном потоке)."""
        if self.client is not None:
            return
        if not host:
            self.log("✖ Укажите IP master")
            return
        self._on_processes_changed()
        name = self.name_edit.text().strip() or socket.gethostname()
        self.client = SlaveClient(host, port, name, self.pool.processes, self.pool)
        self.client.log.connect(self.log)
        self.client.connected.connect(self._on_connected)
        self.client.disconnected.connect(self._on_disconnected)
        self.client.task_started.connect(self._on_task_started)
        self.client.progress.connect(self._on_progress)
        self.client.task_finished.connect(self._on_task_finished)
        self.status_label.setText(f"подключение к {host}:{port}…")
        self._set_connected(True)
        self.client.start()

    def disconnect_from_master(self) -> None:
        if self.client is not None:
            self.client.stop()

    def _on_connected(self, master_name: str, my_name: str) -> None:
        self.hint_label.hide()
        if self.client is not None:
            self.status_label.setText(f"подключён к «{master_name}» ({self.client.host}:"
                                      f"{self.client.port}) как «{my_name}», ожидание задачи")

    def _on_disconnected(self, reason: str) -> None:
        client = self.client
        self.client = None
        if client is not None:
            client.wait(3000)
        self.status_label.setText(f"не подключён ({reason})")
        self.task_label.setText("—")
        self._set_connected(False)

    def _on_task_started(self, task: dict) -> None:
        self.status_label.setText("считает")
        self.task_label.setText(f"функция {task['function']}, N = {task['n']}, слои "
                                f"{task['start']}–{task['end'] - 1} ({task['end'] - task['start']} шт.)")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0 %")

    def _on_progress(self, frac: float) -> None:
        self.progress_bar.setValue(int(frac * 1000))
        self.progress_bar.setFormat(f"{frac * 100:.0f} %")

    def _on_task_finished(self, status: str, partial: float, elapsed: float) -> None:
        if status == "ok":
            self.result_label.setText(f"{partial:.12g} за {elapsed:.3f} с")
            self.status_label.setText("результат отправлен, ожидание задачи")
        elif status == "cancelled":
            self.status_label.setText("задача отменена, ожидание задачи")
        else:
            self.status_label.setText("ошибка расчёта, ожидание задачи")

    # =====================================================================
    # Завершение
    # =====================================================================

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — имя метода Qt
        """Закрыть соединение, остановить потоки и процессы пула."""
        if self.client is not None:
            self.client.stop()
            self.client.wait(15000)
        if self.discovery_thread is not None:
            self.discovery_thread.wait(5000)
        if self._warm_thread is not None:
            self._warm_thread.wait(30000)
        self.pool.close()
        event.accept()
