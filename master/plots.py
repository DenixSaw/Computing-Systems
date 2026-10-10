"""Автоматические графики по CSV-файлам запусков (pandas + matplotlib, бэкенд Agg).

Графики строятся только по серии текущего запуска:
* берутся запуски со status = ok;
* для каждого nodes_count — только последний запуск (наибольший run_id).

Результат в ``runs/series_<название>/``: time_vs_nodes.png, speedup_vs_nodes.png,
efficiency_vs_nodes.png, summary.csv и ``run_<id>/nodes_time.png``.
Функция ничего не бросает: все проблемы возвращаются списком сообщений.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from master.results_store import NODES_FILE, RUNS_FILE

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class PlotResult:
    """Созданные файлы и сообщения для журнала."""

    files: list[Path] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def safe_series_name(series: str) -> str:
    """Название серии, пригодное для имени папки на любой ОС."""
    name = _INVALID_CHARS.sub("_", series).strip(" .")
    return name[:100] or "unnamed"


def series_dir(runs_dir: Path | str, series: str) -> Path:
    return Path(runs_dir) / f"series_{safe_series_name(series)}"


def _read_all(runs_dir: Path, main_name: str, fallback_glob: str):
    """Прочитать основной CSV и запасные файлы в один DataFrame."""
    import pandas as pd

    files = [runs_dir / main_name] if (runs_dir / main_name).exists() else []
    files += sorted(runs_dir.glob(fallback_glob))
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path, dtype={"series": str, "node_name": str,
                                                   "nodes_list": str, "ip": str}))
        except Exception:  # noqa: BLE001 — повреждённый/пустой файл пропускаем
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_series_plots(runs_dir: Path | str, series: str, run_id: int) -> PlotResult:
    """Построить все графики серии ``series`` и диаграмму узлов запуска ``run_id``."""
    result = PlotResult()
    try:
        import matplotlib
        matplotlib.use("Agg")           # без окон: не конфликтует с PyQt
        import pandas  # noqa: F401
    except ImportError as exc:
        result.messages.append(f"⚠ Графики не построены: нет библиотеки ({exc.name}). "
                               f"Установите pandas и matplotlib. CSV сохранены.")
        return result

    runs_dir = Path(runs_dir)
    out_dir = series_dir(runs_dir, series)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result.messages.append(f"⚠ Не удалось создать папку {out_dir}: {exc}")
        return result

    for step in (_series_plots, _run_plot):
        try:
            step(runs_dir, out_dir, series, run_id, result)
        except Exception as exc:  # noqa: BLE001 — ошибка графика не должна ронять приложение
            result.messages.append(f"⚠ Ошибка построения графика ({step.__name__}): {exc!r}")
    return result


# ---------------------------------------------------------------------------
# Графики серии
# ---------------------------------------------------------------------------

def _series_plots(runs_dir: Path, out_dir: Path, series: str, run_id: int,
                  result: PlotResult) -> None:
    import pandas as pd

    df = _read_all(runs_dir, RUNS_FILE, "runs_*.csv")
    if df.empty:
        result.messages.append("Графики серии: в CSV пока нет данных")
        return
    df = df[df["series"].astype(str) == series]
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        result.messages.append(f"Графики серии «{series}»: нет успешных запусков (status = ok)")
        return

    ok["run_id"] = ok["run_id"].astype(int)
    ok["nodes_count"] = ok["nodes_count"].astype(int)
    # Проверка, что серия однородна
    mixed = [col for col in ("function", "N", "split_mode") if ok[col].nunique() > 1]
    if mixed:
        result.messages.append(
            f"⚠ В серии «{series}» есть запуски с разными {', '.join(mixed)} — "
            f"графики серии могут быть некорректны. Дайте сериям разные названия.")

    # По одному (последнему) запуску на каждое число узлов
    latest = (ok.sort_values("run_id").groupby("nodes_count", as_index=False).tail(1)
              .sort_values("nodes_count").reset_index(drop=True))
    last = latest.loc[latest["run_id"].idxmax()]
    subtitle = f"функция: {last['function']}, N = {last['N']}, деление: {last['split_mode']}"

    n = latest["nodes_count"].to_numpy()
    t = latest["total_time_s"].astype(float).to_numpy()

    # 1. Время от числа узлов
    path = out_dir / "time_vs_nodes.png"
    fig, ax = _figure()
    ax.plot(n, t, "o-", color="#1f77b4", linewidth=2, markersize=7)
    for xi, yi in zip(n, t):
        ax.annotate(f"{yi:.2f} с", (xi, yi), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
    ax.set_xlabel("Число узлов")
    ax.set_ylabel("Общее время расчёта, с")
    ax.set_title(f"Серия «{series}»: время расчёта\n{subtitle}", fontsize=11)
    _int_xticks(ax, n)
    ax.set_ylim(bottom=0, top=max(t) * 1.18 if len(t) else 1)
    _save(fig, path, result)

    # 2–3. Ускорение и эффективность
    speedup = [None] * len(latest)
    efficiency = [None] * len(latest)
    if 1 in set(n.tolist()):
        t1 = float(latest.loc[latest["nodes_count"] == 1, "total_time_s"].iloc[0])
        speedup = [t1 / ti if ti > 0 else None for ti in t]
        efficiency = [s / ni if s is not None else None for s, ni in zip(speedup, n)]

        path = out_dir / "speedup_vs_nodes.png"
        fig, ax = _figure()
        ax.plot(n, n, "--", color="gray", label="Идеальное S = n")
        ax.plot(n, speedup, "o-", color="#2ca02c", linewidth=2, markersize=7, label="Фактическое S(n)")
        for xi, si in zip(n, speedup):
            if si is not None:
                ax.annotate(f"{si:.2f}", (xi, si), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=9)
        ax.set_xlabel("Число узлов")
        ax.set_ylabel("Ускорение S(n) = T(1) / T(n)")
        ax.set_title(f"Серия «{series}»: ускорение\n{subtitle}", fontsize=11)
        ax.legend()
        _int_xticks(ax, n)
        ax.set_ylim(bottom=0)
        _save(fig, path, result)

        path = out_dir / "efficiency_vs_nodes.png"
        fig, ax = _figure()
        ax.axhline(1.0, linestyle="--", color="gray", label="Идеальная E = 1")
        ax.plot(n, efficiency, "o-", color="#d62728", linewidth=2, markersize=7, label="E(n) = S(n) / n")
        for xi, ei in zip(n, efficiency):
            if ei is not None:
                ax.annotate(f"{ei:.2f}", (xi, ei), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=9)
        ax.set_xlabel("Число узлов")
        ax.set_ylabel("Эффективность E(n)")
        ax.set_title(f"Серия «{series}»: эффективность\n{subtitle}", fontsize=11)
        ax.legend()
        _int_xticks(ax, n)
        ax.set_ylim(0, max(1.15, max((e for e in efficiency if e is not None), default=1) * 1.15))
        _save(fig, path, result)
    else:
        result.messages.append(
            f"Ускорение и эффективность не построены: в серии «{series}» нет запуска "
            f"на одном узле. Для T(1) выполните расчёт только на master (без slave).")

    # 4. Сводная таблица
    summary = pd.DataFrame({
        "nodes_count": n,
        "run_id": latest["run_id"].to_numpy(),
        "nodes_list": latest["nodes_list"].to_numpy(),
        "total_cores": latest["total_cores"].to_numpy(),
        "total_time_s": t,
        "speedup": speedup,
        "efficiency": efficiency,
    })
    path = out_dir / "summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8")
    result.files.append(path)


# ---------------------------------------------------------------------------
# Диаграмма времени узлов одного запуска
# ---------------------------------------------------------------------------

def _run_plot(runs_dir: Path, out_dir: Path, series: str, run_id: int,
              result: PlotResult) -> None:
    nodes = _read_all(runs_dir, NODES_FILE, "run_nodes_*.csv")
    if nodes.empty:
        return
    nodes = nodes[nodes["run_id"].astype(int) == int(run_id)]
    nodes = nodes[nodes["node_time_s"].notna()]
    if nodes.empty:
        result.messages.append(f"Диаграмма узлов запуска {run_id} не построена: нет времени узлов")
        return

    runs = _read_all(runs_dir, RUNS_FILE, "runs_*.csv")
    total_time = None
    if not runs.empty:
        row = runs[runs["run_id"].astype(int) == int(run_id)]
        if not row.empty:
            total_time = float(row["total_time_s"].iloc[-1])

    names = nodes["node_name"].astype(str).tolist()
    times = nodes["node_time_s"].astype(float).tolist()
    is_master = nodes["is_master"].astype(str).str.lower().eq("true").tolist()
    colors = ["#ff7f0e" if m else "#1f77b4" for m in is_master]
    labels = [f"{nm}\n(master)" if m and "master" not in nm else nm
              for nm, m in zip(names, is_master)]

    run_dir = out_dir / f"run_{int(run_id)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "nodes_time.png"
    fig, ax = _figure(width=max(6.5, 1.3 * len(names) + 2))
    bars = ax.bar(range(len(names)), times, color=colors)
    for bar, value in zip(bars, times):
        ax.annotate(f"{value:.2f} с", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_xticks(range(len(names)), labels, fontsize=9)
    ax.set_ylabel("Время расчёта узла, с")
    total = f"{total_time:.2f} с" if total_time is not None else "—"
    ax.set_title(f"Запуск {run_id} (серия «{series}»): общее время {total}, узлов: {len(names)}",
                 fontsize=11)
    ax.set_ylim(0, max(times) * 1.18 if times else 1)
    _save(fig, path, result)


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------

def _figure(width: float = 7.5, height: float = 5.0):
    """Создать фигуру без pyplot (безопасно вне главного потока)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=110)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, alpha=0.3)
    return fig, ax


def _int_xticks(ax, n) -> None:
    ticks = sorted(set(int(v) for v in n))
    ax.set_xticks(ticks)
    if len(ticks) == 1:
        ax.set_xlim(ticks[0] - 1, ticks[0] + 1)


def _save(fig, path: Path, result: PlotResult) -> None:
    fig.tight_layout()
    fig.savefig(path)
    result.files.append(path)
