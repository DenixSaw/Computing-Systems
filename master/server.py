"""Сервер master: приём подключений slave, раздача долей, локальный расчёт, сборка результата.

Потоки:
* поток приёма подключений (accept) и по одному потоку-обработчику на каждого slave;
* поток heartbeat: рассылает HEARTBEAT и отключает «молчащих» дольше CLIENT_TIMEOUT;
* ``RunWorker`` (QThread) — один расчёт: статическое деление, рассылка TASK, расчёт
  доли master в пуле процессов параллельно со slave, сбор RESULT, пересчёт доли
  отвалившегося slave.

В GUI данные уходят только через сигналы ``pyqtSignal``.
"""

from __future__ import annotations

import math
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from common.config import CLIENT_TIMEOUT, HEARTBEAT_INTERVAL
from common.discovery import DiscoveryResponder
from common.integrator import ComputeCancelled, ComputePool, TaskParams
from common.protocol import (HEARTBEAT, PROGRESS, REGISTER, RESULT, SHUTDOWN, TASK,
                             ProtocolError, close_socket, recv_message, send_message)

MASTER_NODE_ID = 0


# ---------------------------------------------------------------------------
# Подключение одного slave
# ---------------------------------------------------------------------------

@dataclass
class ClientConnection:
    """Состояние подключённого slave."""

    node_id: int
    sock: socket.socket
    ip: str
    port: int
    name: str = "?"
    cores: int = 1
    registered: bool = False
    last_seen: float = field(default_factory=time.monotonic)
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    close_reason: str = ""

    def send(self, msg: dict[str, Any]) -> bool:
        """Отправить сообщение. False — соединение уже не работает."""
        try:
            send_message(self.sock, msg, self.send_lock)
            return True
        except OSError:
            return False

    def close(self, reason: str = "") -> None:
        if reason and not self.close_reason:
            self.close_reason = reason
        close_socket(self.sock)


# ---------------------------------------------------------------------------
# Сервер
# ---------------------------------------------------------------------------

class MasterServer(QObject):
    """TCP-сервер master и UDP-ответчик автопоиска."""

    log = pyqtSignal(str)
    node_connected = pyqtSignal(int, str, str, int)      # node_id, имя, IP, ядра
    node_disconnected = pyqtSignal(int, str, str)        # node_id, имя, причина

    def __init__(self, master_name: str) -> None:
        super().__init__()
        self.master_name = master_name
        self.master_id = uuid.uuid4().hex     # отличает экземпляры master при автопоиске
        self.tcp_port = 0
        self._listen_sock: socket.socket | None = None
        self._clients: dict[int, ClientConnection] = {}
        self._clients_lock = threading.Lock()
        self._next_id = 1
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._responder: DiscoveryResponder | None = None
        # Очередь событий текущего расчёта (RESULT, PROGRESS, отключения)
        self._run_queue: queue.Queue | None = None
        self._run_queue_lock = threading.Lock()

    # --- запуск / остановка ----------------------------------------------------

    def start(self, tcp_port: int, udp_port: int, group: str) -> bool:
        """Открыть TCP-порт и запустить автопоиск. False — TCP-порт занять не удалось."""
        self.stop()
        self._stop.clear()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # SO_REUSEADDR на Windows позволяет двум процессам слушать один порт — там не ставим
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", tcp_port))
            sock.listen(16)
        except OSError as exc:
            sock.close()
            self.log.emit(f"✖ Не удалось открыть TCP-порт {tcp_port}: {exc}. "
                          f"Возможно, он занят другим приложением — выберите другой порт.")
            return False
        sock.settimeout(0.5)
        self._listen_sock = sock
        self.tcp_port = tcp_port
        self.log.emit(f"Сервер слушает TCP-порт {tcp_port}")

        for target, name in ((self._accept_loop, "tcp-accept"), (self._heartbeat_loop, "heartbeat")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

        self._responder = DiscoveryResponder(
            port=udp_port, group=group, master_name=self.master_name, master_id=self.master_id,
            tcp_port_fn=lambda: self.tcp_port, slaves_count_fn=self.slaves_count,
            log=self.log.emit)
        self._responder.start()
        return True

    def stop(self, notify_slaves: bool = False) -> None:
        """Закрыть все сокеты и остановить потоки."""
        self._stop.set()
        if self._responder is not None:
            self._responder.stop()
            self._responder = None
        close_socket(self._listen_sock)
        self._listen_sock = None
        with self._clients_lock:
            clients = list(self._clients.values())
        for c in clients:
            if notify_slaves:
                c.send({"type": SHUTDOWN, "reason": "exit"})
            c.close("сервер остановлен")
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()

    # --- доступ к узлам ----------------------------------------------------------

    def slaves_count(self) -> int:
        with self._clients_lock:
            return sum(1 for c in self._clients.values() if c.registered)

    def slaves(self) -> list[ClientConnection]:
        """Снимок списка зарегистрированных slave."""
        with self._clients_lock:
            return [c for c in self._clients.values() if c.registered]

    def get_client(self, node_id: int) -> ClientConnection | None:
        with self._clients_lock:
            return self._clients.get(node_id)

    def set_run_queue(self, q: queue.Queue | None) -> None:
        """Направлять события slave (RESULT/PROGRESS/отключение) в очередь текущего расчёта."""
        with self._run_queue_lock:
            self._run_queue = q

    def _post(self, event: tuple) -> None:
        with self._run_queue_lock:
            q = self._run_queue
        if q is not None:
            q.put(event)

    # --- потоки -------------------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            sock = self._listen_sock
            if sock is None:
                break
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._clients_lock:
                node_id = self._next_id
                self._next_id += 1
                client = ClientConnection(node_id=node_id, sock=conn, ip=addr[0], port=addr[1])
                self._clients[node_id] = client
            self.log.emit(f"Входящее подключение от {addr[0]}:{addr[1]}")
            t = threading.Thread(target=self._handle_client, args=(client,),
                                 name=f"client-{node_id}", daemon=True)
            t.start()

    def _unique_name(self, name: str) -> str:
        """Сделать имя узла уникальным (суффикс #2, #3, …)."""
        with self._clients_lock:
            taken = {c.name for c in self._clients.values() if c.registered}
        taken.add(self.master_name)
        if name not in taken:
            return name
        k = 2
        while f"{name}#{k}" in taken:
            k += 1
        return f"{name}#{k}"

    def _handle_client(self, c: ClientConnection) -> None:
        """Поток-обработчик одного slave: регистрация и приём сообщений."""
        reason = "соединение закрыто"
        try:
            c.sock.settimeout(10.0)
            msg = recv_message(c.sock)
            if msg is None or msg.get("type") != REGISTER:
                raise ProtocolError("ожидалось сообщение REGISTER")
            c.sock.settimeout(None)
            c.name = self._unique_name(str(msg.get("name") or c.ip)[:64])
            c.cores = max(1, int(msg.get("cores", 1)))
            c.last_seen = time.monotonic()
            c.send({"type": REGISTER, "status": "ok", "node_id": c.node_id,
                    "name": c.name, "master_name": self.master_name})
            c.registered = True
            self.log.emit(f"Узел «{c.name}» ({c.ip}) зарегистрирован, ядер: {c.cores}")
            self.node_connected.emit(c.node_id, c.name, c.ip, c.cores)

            while not self._stop.is_set():
                msg = recv_message(c.sock)
                if msg is None:
                    reason = "slave закрыл соединение"
                    break
                c.last_seen = time.monotonic()
                mtype = msg.get("type")
                if mtype in (PROGRESS, RESULT):
                    self._post((mtype, c.node_id, msg))
                elif mtype == SHUTDOWN:
                    reason = "slave завершил работу"
                    break
                # HEARTBEAT: достаточно обновить last_seen
        except socket.timeout:
            reason = "не прислал REGISTER вовремя"
        except (ConnectionError, OSError) as exc:
            reason = f"обрыв соединения ({exc.__class__.__name__})"
        except (ProtocolError, ValueError, TypeError) as exc:
            reason = f"ошибка протокола: {exc}"
        finally:
            if c.close_reason:
                reason = c.close_reason
            c.close()
            with self._clients_lock:
                self._clients.pop(c.node_id, None)
            if c.registered:
                self.log.emit(f"Узел «{c.name}» ({c.ip}) отключён: {reason}")
                self.node_disconnected.emit(c.node_id, c.name, reason)
                self._post(("DISCONNECTED", c.node_id, {"reason": reason}))
            elif not self._stop.is_set():
                self.log.emit(f"Подключение {c.ip}:{c.port} закрыто: {reason}")

    def _heartbeat_loop(self) -> None:
        """Периодический HEARTBEAT и отключение узлов, молчащих дольше CLIENT_TIMEOUT."""
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            now = time.monotonic()
            for c in self.slaves():
                if now - c.last_seen > CLIENT_TIMEOUT:
                    self.log.emit(f"⚠ Узел «{c.name}» не отвечает {CLIENT_TIMEOUT:.0f} с — отключаю")
                    c.close(f"таймаут {CLIENT_TIMEOUT:.0f} с без сообщений")
                elif not c.send({"type": HEARTBEAT, "t": time.time()}):
                    c.close("ошибка отправки HEARTBEAT")


# ---------------------------------------------------------------------------
# Один расчёт
# ---------------------------------------------------------------------------

@dataclass
class Assignment:
    """Доля задачи одного узла."""

    node_id: int
    name: str
    ip: str
    cores: int
    is_master: bool
    start: int
    end: int                 # не включительно
    partial: float | None = None
    node_time: float | None = None
    done: bool = False
    recovered: bool = False  # долю пересчитал master

    @property
    def layers(self) -> int:
        return self.end - self.start


class RunWorker(QThread):
    """Выполняет один распределённый расчёт (вне GUI-потока)."""

    log = pyqtSignal(str)
    node_status = pyqtSignal(int, str)             # node_id, статус
    node_progress = pyqtSignal(int, float)         # node_id, 0..1
    node_done = pyqtSignal(int, float, float)      # node_id, частичная сумма, время узла
    node_chunks = pyqtSignal(int, list)            # кто посчитал, [[start, профиль, вклад], …]
    run_finished = pyqtSignal(dict)                # итог расчёта

    def __init__(self, server: MasterServer, pool: ComputePool, params: TaskParams,
                 assignments: list[Assignment], t_start: float) -> None:
        super().__init__()
        self.server = server
        self.pool = pool
        self.params = params
        self.assignments = assignments
        self.t_start = t_start              # момент нажатия «Запустить» (perf_counter)
        self.run_tag = uuid.uuid4().hex[:12]
        self._cancel = threading.Event()
        self._events: queue.Queue = queue.Queue()
        self._local_jobs: queue.Queue = queue.Queue()

    def cancel(self) -> None:
        """Запросить остановку расчёта."""
        self._cancel.set()

    # --- локальный расчёт master ----------------------------------------------------

    def _local_loop(self) -> None:
        """Поток локального расчёта: своя доля master и (при отказе) доли slave."""
        while True:
            a: Assignment | None = self._local_jobs.get()
            if a is None:
                return
            try:
                partial, elapsed = self.pool.compute(
                    self.params, a.start, a.end,
                    progress_cb=lambda f, nid=a.node_id: self.node_progress.emit(nid, f),
                    cancel_event=self._cancel,
                    # слои, посчитанные master (в т.ч. при пересчёте), красятся цветом master
                    chunk_cb=lambda s, prof, part: self.node_chunks.emit(
                        MASTER_NODE_ID, [[s, prof, part]]))
                self._events.put(("LOCAL_DONE", a.node_id, {"partial": partial, "time": elapsed}))
            except ComputeCancelled:
                self._events.put(("LOCAL_CANCELLED", a.node_id, {}))
                return
            except Exception as exc:  # noqa: BLE001 — сообщаем об ошибке в основной цикл
                self._events.put(("LOCAL_ERROR", a.node_id, {"error": repr(exc)}))
                return

    # --- основной цикл ----------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 — линейная логика одного расчёта
        by_id = {a.node_id: a for a in self.assignments}
        warnings: list[str] = []
        status = "ok"
        error = ""
        self.server.set_run_queue(self._events)
        local_thread = threading.Thread(target=self._local_loop, name="master-compute", daemon=True)
        try:
            # 1. Каждому slave — одно сообщение TASK с диапазоном и параметрами
            not_sent: list[Assignment] = []
            for a in self.assignments:
                if a.is_master:
                    continue
                client = self.server.get_client(a.node_id)
                msg = {"type": TASK, "run_tag": self.run_tag, "start": a.start, "end": a.end,
                       "params": self.params.to_dict()}
                if client is not None and client.send(msg):
                    self.node_status.emit(a.node_id, "считает")
                else:
                    not_sent.append(a)

            # 2. Сразу же master считает свою долю (параллельно со slave)
            master = by_id.get(MASTER_NODE_ID)
            if master is not None:
                self.node_status.emit(MASTER_NODE_ID, "считает")
                self._local_jobs.put(master)
            for a in not_sent:
                self._fail_over(a, "не удалось отправить задачу", warnings)
                status = "recovered"
            local_thread.start()
            self.log.emit(f"Задачи разосланы, master считает свою долю "
                          f"({self.pool.processes} процессов)")

            # 3. Сбор результатов
            while not all(a.done for a in self.assignments):
                if self._cancel.is_set():
                    status = "stopped"
                    break
                try:
                    kind, node_id, data = self._events.get(timeout=0.2)
                except queue.Empty:
                    continue
                a = by_id.get(node_id)
                if a is None:
                    continue
                if kind == PROGRESS and data.get("run_tag") == self.run_tag and not a.recovered:
                    self.node_progress.emit(node_id, float(data.get("progress", 0.0)))
                    chunks = data.get("chunks")
                    if isinstance(chunks, list) and chunks:
                        self.node_chunks.emit(node_id, chunks)
                elif kind == RESULT and data.get("run_tag") == self.run_tag and not a.recovered:
                    if data.get("error"):
                        self._fail_over(a, f"ошибка на узле: {data['error']}", warnings)
                        status = "recovered"
                        continue
                    a.partial = float(data["partial_sum"])
                    a.node_time = float(data["node_time_s"])
                    a.done = True
                    self.node_progress.emit(node_id, 1.0)
                    self.node_status.emit(node_id, "готово")
                    self.node_done.emit(node_id, a.partial, a.node_time)
                    self.log.emit(f"Результат от «{a.name}»: {a.partial:.12g} за {a.node_time:.3f} с")
                elif kind == "DISCONNECTED" and not a.done and not a.recovered:
                    self._fail_over(a, data.get("reason", "отключился"), warnings)
                    status = "recovered"
                elif kind == "LOCAL_DONE":
                    a.partial = float(data["partial"])
                    a.node_time = float(data["time"])
                    a.done = True
                    self.node_progress.emit(node_id, 1.0)
                    if a.recovered:
                        self.node_status.emit(node_id, "пересчитано master")
                        self.log.emit(f"Master пересчитал долю «{a.name}» за {a.node_time:.3f} с")
                    else:
                        self.node_status.emit(node_id, "готово")
                        self.log.emit(f"Master посчитал свою долю: {a.partial:.12g} "
                                      f"за {a.node_time:.3f} с")
                    self.node_done.emit(node_id, a.partial, a.node_time)
                elif kind == "LOCAL_ERROR":
                    status = "failed"
                    error = f"ошибка локального расчёта: {data.get('error')}"
                    break
                elif kind == "LOCAL_CANCELLED":
                    status = "stopped"
                    break
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = repr(exc)
        finally:
            self.server.set_run_queue(None)
            if status in ("stopped", "failed"):
                self._cancel.set()
                for a in self.assignments:
                    if not a.is_master and not a.done and not a.recovered:
                        client = self.server.get_client(a.node_id)
                        if client is not None:
                            client.send({"type": SHUTDOWN, "reason": "cancel",
                                         "run_tag": self.run_tag})
                        self.node_status.emit(a.node_id, "остановлен")
                if master_assignment := by_id.get(MASTER_NODE_ID):
                    if not master_assignment.done:
                        self.node_status.emit(MASTER_NODE_ID, "остановлен")
            self._local_jobs.put(None)
            if local_thread.is_alive():
                local_thread.join(timeout=10)

        total_time = time.perf_counter() - self.t_start
        integral = None
        if status in ("ok", "recovered"):
            integral = math.fsum(a.partial or 0.0 for a in self.assignments)
        self.run_finished.emit({
            "status": status,
            "error": error,
            "integral": integral,
            "total_time": total_time,
            "warnings": warnings,
            "assignments": self.assignments,
        })

    def _fail_over(self, a: Assignment, reason: str, warnings: list[str]) -> None:
        """Slave не справился: master пересчитывает его долю сам."""
        a.recovered = True
        text = (f"⚠ Узел «{a.name}» ({a.ip}) выбыл из расчёта: {reason}. "
                f"Master пересчитает его долю (слои {a.start}–{a.end - 1}) сам.")
        warnings.append(text)
        self.log.emit(text)
        self.node_status.emit(a.node_id, "пересчёт на master")
        self.node_progress.emit(a.node_id, 0.0)
        self._local_jobs.put(a)
