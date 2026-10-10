"""Общие константы приложения: порты, таймауты, параметры задачи по умолчанию."""

from __future__ import annotations

import math
import os
from pathlib import Path

# --- Сеть -------------------------------------------------------------------
TCP_PORT: int = 5000            # порт master для подключения slave
UDP_PORT: int = 5001            # порт UDP-автопоиска
DEFAULT_GROUP: str = "lab"      # код группы по умолчанию
PROTOCOL_VERSION: int = 1       # версия протокола (проверяется при автопоиске)

MAX_MESSAGE_SIZE: int = 16 * 1024 * 1024   # защита от «мусорной» длины сообщения
CONNECT_TIMEOUT: float = 5.0               # таймаут TCP-подключения slave к master, с
HEARTBEAT_INTERVAL: float = 2.0            # период отправки HEARTBEAT, с
CLIENT_TIMEOUT: float = 15.0               # тишина от узла дольше этого = узел потерян, с

DISCOVERY_ATTEMPTS: int = 3                # сколько раз повторять DISCOVER
DISCOVERY_PAUSE: float = 0.4               # пауза между повторами, с
DISCOVERY_WINDOW: float = 2.5              # сколько всего ждать ответов, с

PROGRESS_MIN_INTERVAL: float = 0.3         # как часто slave шлёт PROGRESS, с

# --- Задача -----------------------------------------------------------------
FUNCTIONS: dict[str, str] = {
    "test": "Тестовая: sin(x)·cos(y)·exp(z)",
    "heavy": "Тяжёлая: sin(xy)·e^(−z) + √(x²+y²+z²+1)·cos(x+yz) + ln(1+xyz)",
}

# Границы области по умолчанию для каждой функции: (ax, bx, ay, by, az, bz)
DEFAULT_BOUNDS: dict[str, tuple[float, float, float, float, float, float]] = {
    "test": (0.0, math.pi, 0.0, math.pi / 2, 0.0, 1.0),
    "heavy": (0.0, 2.0, 0.0, 2.0, 0.0, 2.0),
}

N_QUICK: int = 200          # быстрый тест (доли секунды)
N_DEMO: int = 1200          # демонстрация: тяжёлая функция ≈ 2 мин на одном ядре (~70 нс/точку)
N_DEFAULT: int = N_DEMO
N_MAX: int = 4000

# Сколько точек сетки обрабатывать за один вызов NumPy (ограничивает память)
CHUNK_POINTS: int = 4_000_000

SPLIT_MODES: dict[str, str] = {
    "cores": "По числу ядер",
    "equal": "Равномерно",
}

# --- Пути -------------------------------------------------------------------
APP_DIR: Path = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR: Path = APP_DIR / "runs"


def cpu_count() -> int:
    """Число доступных ядер (не меньше 1)."""
    return os.cpu_count() or 1
