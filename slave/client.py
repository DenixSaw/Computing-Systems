"""Клиент slave: подключение к master, получение доли, расчёт в пуле процессов, отправка результата.

Сеть работает в ``SlaveClient`` (QThread), расчёт — в отдельном потоке, который
управляет пулом процессов; HEARTBEAT шлёт ещё один поток. В GUI — только сигналы.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from common.config import CONNECT_TIMEOUT, HEARTBEAT_INTERVAL, PROGRESS_MIN_INTERVAL
from common.integrator import ComputeCancelled, ComputePool, TaskParams, validate_params
from common.protocol import (HEARTBEAT, PROGRESS, REGISTER, RESULT, SHUTDOWN, TASK,
                             ProtocolError, close_socket, recv_message, send_message)


class SlaveClient(QThread):
    """Соединение slave с master (вне GUI-потока)."""

    log = pyqtSignal(str)
    connected = pyqtSignal(str, str)          # имя master, имя этого узла (после дедупликации)
    disconnected = pyqtSignal(str)            # причина
    task_started = pyqtSignal(dict)           # описание задачи
    progress = pyqtSignal(float)              # 0..1
    task_finished = pyqtSignal(str, float, float)   # статус (ok/cancelled/error), частичная сумма, время

    def __init__(self, host: str, port: int, name: str, cores: int, pool: ComputePool) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.name = name
        self.cores = cores
        self.pool = pool
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._compute_thread: threading.Thread | None = None

    # --- управление -----------------------------------------------------------------

    def stop(self) -> None:
        """Отключиться от master (вызывается из GUI)."""
        self._stop.set()
        self._cancel.set()
        sock = self._sock
        if sock is not None:
            try:
                send_message(sock, {"type": SHUTDOWN, "reason": "slave exit"}, self._send_lock)
            except OSError:
                pass
            close_socket(sock)

    def _send(self, msg: dict[str, Any]) -> bool:
        sock = self._sock
        if sock is None:
            return False
        try:
            send_message(sock, msg, self._send_lock)
            return True
        except OSError:
            return False

    # --- основной поток -------------------------------------------------------------------

    def run(self) -> None:
        reason = "соединение закрыто"
        try:
            self.log.emit(f"Подключение к master {self.host}:{self.port} …")
            try:
                sock = socket.create_connection((self.host, self.port), timeout=CONNECT_TIMEOUT)
            except socket.timeout:
                raise ConnectionError(f"master {self.host}:{self.port} не ответил за "
                                      f"{CONNECT_TIMEOUT:.0f} с (проверьте IP и файрвол)") from None
            except ConnectionRefusedError:
                raise ConnectionError(f"в подключении отказано: на {self.host}:{self.port} "
                                      f"master не запущен или порт другой") from None
            except socket.gaierror:
                raise ConnectionError(f"неизвестный адрес «{self.host}»") from None
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock

            send_message(sock, {"type": REGISTER, "name": self.name, "cores": self.cores},
                         self._send_lock)
            sock.settimeout(10.0)
            reply = recv_message(sock)
            if reply is None or reply.get("type") != REGISTER or reply.get("status") != "ok":
                raise ProtocolError("master не подтвердил регистрацию")
            sock.settimeout(None)
            my_name = str(reply.get("name", self.name))
            self.log.emit(f"Зарегистрирован у master «{reply.get('master_name')}» как «{my_name}»")
            self.connected.emit(str(reply.get("master_name", "?")), my_name)

            threading.Thread(target=self._heartbeat_loop, name="slave-heartbeat",
                             daemon=True).start()

            while not self._stop.is_set():
                msg = recv_message(sock)
                if msg is None:
                    reason = "master закрыл соединение"
                    break
                mtype = msg.get("type")
                if mtype == TASK:
                    self._start_task(msg)
                elif mtype == SHUTDOWN:
                    if msg.get("reason") == "cancel":
                        self.log.emit("Master остановил расчёт — отменяю текущую задачу")
                        self._cancel.set()
                    else:
                        reason = "master завершил работу"
                        break
                # HEARTBEAT от master: ничего не делаем
        except socket.timeout:
            reason = "master не ответил на регистрацию"
        except ConnectionError as exc:
            reason = str(exc) if self._sock is None else f"обрыв соединения ({exc.__class__.__name__})"
        except ProtocolError as exc:
            reason = f"ошибка протокола: {exc}"
        except OSError as exc:
            reason = "отключено пользователем" if self._stop.is_set() else f"ошибка сети: {exc}"
        finally:
            if self._stop.is_set():
                reason = "отключено пользователем"
            self._stop.set()
            self._cancel.set()
            close_socket(self._sock)
            if self._compute_thread is not None:
                self._compute_thread.join(timeout=10)
            self._sock = None
            self.log.emit(f"Соединение с master закрыто: {reason}")
            self.disconnected.emit(reason)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            if not self._send({"type": HEARTBEAT, "t": time.time()}):
                break

    # --- расчёт задачи ---------------------------------------------------------------------

    def _start_task(self, msg: dict[str, Any]) -> None:
        """Получен TASK: запустить расчёт в отдельном потоке (приём сообщений не блокируется)."""
        if self._compute_thread is not None and self._compute_thread.is_alive():
            self.log.emit("⚠ Новая задача пришла до завершения предыдущей — предыдущая отменяется")
            self._cancel.set()
            self._compute_thread.join()
        self._cancel = threading.Event()
        self._compute_thread = threading.Thread(target=self._compute, args=(msg, self._cancel),
                                                name="slave-compute", daemon=True)
        self._compute_thread.start()

    def _compute(self, msg: dict[str, Any], cancel: threading.Event) -> None:
        run_tag = msg.get("run_tag")
        try:
            params = TaskParams.from_dict(msg["params"])
            start, end = int(msg["start"]), int(msg["end"])
        except (KeyError, TypeError, ValueError) as exc:
            self._send({"type": RESULT, "run_tag": run_tag, "error": f"некорректная задача: {exc}"})
            return
        err = validate_params(params)
        if err:
            self._send({"type": RESULT, "run_tag": run_tag, "error": err})
            return

        self.task_started.emit({"function": params.function, "n": params.n,
                                "start": start, "end": end})
        self.log.emit(f"Получена задача: функция {params.function}, N={params.n}, "
                      f"слои {start}–{end - 1} ({end - start} шт.), процессов: {self.pool.processes}")
        last_sent = 0.0

        def on_progress(frac: float) -> None:
            nonlocal last_sent
            self.progress.emit(frac)
            now = time.monotonic()
            if now - last_sent >= PROGRESS_MIN_INTERVAL or frac >= 1.0:
                last_sent = now
                self._send({"type": PROGRESS, "run_tag": run_tag, "progress": frac})

        try:
            partial, elapsed = self.pool.compute(params, start, end, on_progress, cancel)
        except ComputeCancelled:
            self.log.emit("Задача отменена")
            self.task_finished.emit("cancelled", 0.0, 0.0)
            return
        except Exception as exc:  # noqa: BLE001 — сообщаем master об ошибке
            self.log.emit(f"✖ Ошибка расчёта: {exc!r}")
            self._send({"type": RESULT, "run_tag": run_tag, "error": repr(exc)})
            self.task_finished.emit("error", 0.0, 0.0)
            return

        sent = self._send({"type": RESULT, "run_tag": run_tag,
                           "partial_sum": partial, "node_time_s": elapsed})
        self.log.emit(f"Расчёт завершён за {elapsed:.3f} с, частичная сумма {partial:.12g}"
                      + ("" if sent else " (⚠ не удалось отправить master)"))
        self.task_finished.emit("ok", partial, elapsed)
