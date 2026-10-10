"""UDP-автопоиск master в локальной сети (DISCOVER / ANNOUNCE).

* Master запускает ``DiscoveryResponder``: слушает UDP-порт и на запрос ``DISCOVER``
  с совпадающим кодом группы отвечает unicast-сообщением ``ANNOUNCE``.
* Slave вызывает ``discover_masters()``: рассылает ``DISCOVER`` широковещательно
  (255.255.255.255 и broadcast-адреса всех локальных интерфейсов) несколько раз
  и собирает ответы в течение заданного окна времени.

Только стандартный модуль ``socket``. Модуль не зависит от PyQt: журнал
передаётся колбэком ``log``.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Callable

from common.config import (DISCOVERY_ATTEMPTS, DISCOVERY_PAUSE, DISCOVERY_WINDOW,
                           PROTOCOL_VERSION, UDP_PORT)
from common.protocol import ANNOUNCE, DISCOVER

LogFn = Callable[[str], None]


def _encode(msg: dict[str, Any]) -> bytes:
    return json.dumps(msg, ensure_ascii=False).encode("utf-8")


def _decode(data: bytes) -> dict[str, Any] | None:
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return msg if isinstance(msg, dict) else None


# ---------------------------------------------------------------------------
# Локальные адреса
# ---------------------------------------------------------------------------

def _route_ip(target: str) -> str | None:
    """IP локального интерфейса, через который ОС отправила бы пакет на ``target``.

    UDP ``connect`` ничего не отправляет в сеть, только выбирает маршрут.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 9))
        ip = s.getsockname()[0]
        return None if ip.startswith("0.") else ip
    except OSError:
        return None
    finally:
        s.close()


def local_ipv4_addresses() -> list[str]:
    """Все IPv4-адреса этой машины (без 127.x), определённые средствами ``socket``."""
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    for target in ("8.8.8.8", "10.255.255.255", "192.168.255.255", "172.31.255.255"):
        ip = _route_ip(target)
        if ip:
            ips.add(ip)
    return sorted(ip for ip in ips if not ip.startswith("127.") and not ip.startswith("169.254."))


def primary_ip() -> str:
    """Основной IP машины (для отображения пользователю)."""
    return _route_ip("8.8.8.8") or (local_ipv4_addresses() or ["127.0.0.1"])[0]


def broadcast_addresses() -> list[str]:
    """Адреса для рассылки: 255.255.255.255, x.y.z.255 для каждого локального IPv4 (/24)
    и 127.0.0.1 — чтобы находился master, запущенный на этой же машине."""
    result = ["255.255.255.255"]
    for ip in local_ipv4_addresses():
        parts = ip.split(".")
        result.append(".".join(parts[:3] + ["255"]))
    result.append("127.0.0.1")
    # убираем дубли, сохраняя порядок
    return list(dict.fromkeys(result))


# ---------------------------------------------------------------------------
# Master: ответчик
# ---------------------------------------------------------------------------

class DiscoveryResponder:
    """Поток master, отвечающий на DISCOVER сообщением ANNOUNCE."""

    def __init__(self, port: int, group: str, master_name: str, master_id: str,
                 tcp_port_fn: Callable[[], int], slaves_count_fn: Callable[[], int],
                 log: LogFn) -> None:
        self.port = port
        self.group = group
        self.master_name = master_name
        self.master_id = master_id      # уникален для экземпляра master
        self._tcp_port_fn = tcp_port_fn
        self._slaves_count_fn = slaves_count_fn
        self._log = log
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Один поиск slave = несколько одинаковых запросов (повторы × интерфейсы).
        # Отвечаем на каждый, а в журнал пишем один раз на request_id.
        self._logged_requests: dict[str, None] = {}

    def start(self) -> bool:
        """Открыть UDP-сокет и запустить поток. False — порт занят (автопоиск не работает)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self.port))
        except OSError as exc:
            sock.close()
            self._log(f"⚠ UDP-порт {self.port} занят или недоступен ({exc}). "
                      f"Автопоиск не работает, но ручное подключение по TCP доступно.")
            return False
        sock.settimeout(0.5)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="udp-discovery", daemon=True)
        self._thread.start()
        self._log(f"Автопоиск: слушаю UDP-порт {self.port}, код группы «{self.group}»")
        return True

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                # На Windows recvfrom может вернуть ConnectionResetError после ICMP — игнорируем
                if self._stop.is_set():
                    break
                continue
            msg = _decode(data)
            if not msg or msg.get("type") != DISCOVER:
                continue
            group = str(msg.get("group", ""))
            request_id = str(msg.get("request_id", f"{addr[0]}:{addr[1]}"))
            first = request_id not in self._logged_requests
            if first:
                self._logged_requests[request_id] = None
                while len(self._logged_requests) > 200:
                    self._logged_requests.pop(next(iter(self._logged_requests)))
                self._log(f"Автопоиск: получен запрос DISCOVER от {addr[0]}:{addr[1]} "
                          f"(группа «{group}»)")
            if group != self.group:
                if first:
                    self._log("Автопоиск: код группы не совпадает — запрос проигнорирован")
                continue
            if int(msg.get("version", -1)) != PROTOCOL_VERSION:
                if first:
                    self._log("Автопоиск: другая версия протокола — запрос проигнорирован")
                continue
            reply = {
                "type": ANNOUNCE,
                "version": PROTOCOL_VERSION,
                "group": self.group,
                "name": self.master_name,
                "master_id": self.master_id,
                "ip": _route_ip(addr[0]) or primary_ip(),
                "tcp_port": self._tcp_port_fn(),
                "slaves": self._slaves_count_fn(),
                "request_id": msg.get("request_id"),
            }
            try:
                self._sock.sendto(_encode(reply), addr)
                if first:
                    self._log(f"Автопоиск: отправлен ANNOUNCE на {addr[0]}:{addr[1]} "
                              f"(IP {reply['ip']}, TCP {reply['tcp_port']})")
            except OSError as exc:
                self._log(f"Автопоиск: не удалось отправить ответ {addr[0]}: {exc}")

    def stop(self) -> None:
        """Остановить поток и закрыть сокет."""
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sock = None
        self._thread = None


# ---------------------------------------------------------------------------
# Slave: поиск
# ---------------------------------------------------------------------------

def discover_masters(group: str, port: int = UDP_PORT, log: LogFn | None = None,
                     attempts: int = DISCOVERY_ATTEMPTS, pause: float = DISCOVERY_PAUSE,
                     window: float = DISCOVERY_WINDOW,
                     stop_event: threading.Event | None = None) -> list[dict[str, Any]]:
    """Найти master в локальной сети. Блокирующая функция (вызывать не из GUI-потока).

    Возвращает список словарей: name, ip (лучший адрес), all_ips, tcp_port, slaves, group.
    """
    log = log or (lambda _m: None)
    request_id = f"{time.time_ns():x}"
    request = _encode({"type": DISCOVER, "version": PROTOCOL_VERSION,
                       "group": group, "request_id": request_id})
    targets = broadcast_addresses()
    found: dict[tuple[str, int], dict[str, Any]] = {}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", 0))
        sock.settimeout(0.2)
        log(f"Поиск master: группа «{group}», UDP-порт {port}, адреса: {', '.join(targets)}")

        start = time.monotonic()
        sent = 0
        next_send = start
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            now = time.monotonic()
            if sent < attempts and now >= next_send:
                for addr in targets:
                    try:
                        sock.sendto(request, (addr, port))
                    except OSError:
                        pass  # интерфейс может не поддерживать broadcast — пропускаем
                sent += 1
                next_send = now + pause
            if now - start >= window:
                break
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                continue
            msg = _decode(data)
            if not msg or msg.get("type") != ANNOUNCE or msg.get("group") != group:
                continue
            tcp_port = int(msg.get("tcp_port", 0))
            # Один master отвечает через каждый сетевой интерфейс (Wi-Fi, Ethernet, VPN,
            # виртуальные адаптеры), поэтому ответы группируем по его уникальному id
            key = str(msg.get("master_id") or f"{msg.get('name')}:{tcp_port}")
            ips = {str(msg.get("ip") or addr[0]), addr[0]}
            if key not in found:
                found[key] = {
                    "name": str(msg.get("name", "?")),
                    "ips": set(),
                    "tcp_port": tcp_port,
                    "slaves": int(msg.get("slaves", 0)),
                    "group": group,
                }
                log(f"Найден master «{found[key]['name']}» ({addr[0]}:{tcp_port})")
            found[key]["ips"] |= ips
            found[key]["slaves"] = int(msg.get("slaves", 0))
    finally:
        sock.close()

    my_ip = primary_ip()
    masters = []
    for m in found.values():
        ips = sorted(m.pop("ips"))
        m["ip"] = _best_ip(ips, my_ip)
        m["all_ips"] = ips
        masters.append(m)
    log(f"Поиск завершён: найдено master — {len(masters)}")
    return masters


def _best_ip(ips: list[str], my_ip: str) -> str:
    """Выбрать адрес master для подключения: совпадающий с нашим, из той же сети /24,
    затем любой не-loopback, в крайнем случае 127.0.0.1."""
    if my_ip in ips:
        return my_ip                      # master на этой же машине
    prefix = my_ip.rsplit(".", 1)[0] + "."
    same_net = [ip for ip in ips if ip.startswith(prefix)]
    if same_net:
        return same_net[0]
    real = [ip for ip in ips if not ip.startswith("127.")]
    return real[0] if real else ips[0]
