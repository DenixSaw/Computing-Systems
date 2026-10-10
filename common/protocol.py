"""Протокол обмена сообщениями по TCP.

Формат кадра: 4 байта длины (big-endian, беззнаковое целое) + JSON в UTF-8.
Каждое сообщение — словарь с обязательным полем ``type``.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import Any

from common.config import MAX_MESSAGE_SIZE

# Типы сообщений TCP
REGISTER = "REGISTER"     # slave -> master: имя, ядра; master -> slave: подтверждение
TASK = "TASK"             # master -> slave: диапазон слоёв и параметры задачи
PROGRESS = "PROGRESS"     # slave -> master: процент выполнения
RESULT = "RESULT"         # slave -> master: частичная сумма и время расчёта
HEARTBEAT = "HEARTBEAT"   # в обе стороны: «я жив»
SHUTDOWN = "SHUTDOWN"     # master -> slave: reason="cancel" (отмена задачи) или "exit"

# Типы сообщений UDP-автопоиска
DISCOVER = "DISCOVER"     # slave -> broadcast
ANNOUNCE = "ANNOUNCE"     # master -> slave (unicast)

_HEADER = struct.Struct(">I")


class ProtocolError(Exception):
    """Нарушение формата сообщения."""


def encode_message(msg: dict[str, Any]) -> bytes:
    """Упаковать словарь в кадр «длина + JSON»."""
    payload = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_MESSAGE_SIZE:
        raise ProtocolError(f"Сообщение слишком большое: {len(payload)} байт")
    return _HEADER.pack(len(payload)) + payload


def decode_payload(payload: bytes) -> dict[str, Any]:
    """Разобрать JSON-полезную нагрузку кадра."""
    try:
        msg = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Некорректный JSON: {exc}") from exc
    if not isinstance(msg, dict) or "type" not in msg:
        raise ProtocolError("Сообщение должно быть объектом с полем 'type'")
    return msg


def send_message(sock: socket.socket, msg: dict[str, Any],
                 lock: threading.Lock | None = None) -> None:
    """Отправить сообщение целиком. ``lock`` защищает сокет при записи из разных потоков."""
    data = encode_message(msg)
    if lock is None:
        sock.sendall(data)
    else:
        with lock:
            sock.sendall(data)


def recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Прочитать ровно ``n`` байт (recv может вернуть меньше, поэтому читаем в цикле).

    Возвращает ``None``, если соединение закрыто до начала чтения,
    и бросает ``ConnectionError``, если оно оборвалось посередине.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            if not buf:
                return None
            raise ConnectionError("Соединение оборвалось посреди сообщения")
        buf.extend(chunk)
    return bytes(buf)


def recv_message(sock: socket.socket) -> dict[str, Any] | None:
    """Принять одно сообщение. ``None`` — собеседник корректно закрыл соединение."""
    header = recv_exact(sock, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > MAX_MESSAGE_SIZE:
        raise ProtocolError(f"Слишком длинное сообщение: {length} байт")
    payload = recv_exact(sock, length) if length else b""
    if payload is None:
        raise ConnectionError("Соединение оборвалось после заголовка")
    return decode_payload(payload)


def close_socket(sock: socket.socket | None) -> None:
    """Аккуратно закрыть сокет (shutdown будит потоки, заблокированные в recv)."""
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass
