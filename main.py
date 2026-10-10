"""Точка входа: стартовое окно с выбором роли (Master / Slave).

Важно для Windows и macOS (метод spawn): процессы пула заново импортируют этот
модуль, поэтому на верхнем уровне нет тяжёлых импортов (PyQt) и запуск GUI
защищён ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    """Показать стартовое окно и открыть окно выбранной роли."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout,
                                 QWidget)

    app = QApplication(sys.argv)
    app.setApplicationName("Master–Slave Simpson")

    windows: list[QMainWindow] = []   # держим ссылку на окно роли, чтобы его не собрал GC

    class StartWindow(QWidget):
        """Небольшое окно с двумя крупными кнопками выбора роли."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Распределённые вычисления: Master–Slave")
            self.setFixedSize(460, 300)
            layout = QVBoxLayout(self)
            title = QLabel("Тройной интеграл методом Симпсона\nраспределённо по схеме Master–Slave")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = title.font()
            font.setPointSize(font.pointSize() + 3)
            font.setBold(True)
            title.setFont(font)
            layout.addWidget(title)
            for text, role in (("Запустить как Master", "master"), ("Запустить как Slave", "slave")):
                btn = QPushButton(text)
                btn.setMinimumHeight(70)
                f = btn.font()
                f.setPointSize(f.pointSize() + 4)
                btn.setFont(f)
                btn.clicked.connect(lambda _checked=False, r=role: self.open_role(r))
                layout.addWidget(btn)

        def open_role(self, role: str) -> None:
            # Импорт здесь: окно роли тянет сеть, NumPy и matplotlib только при выборе
            if role == "master":
                from master.gui_master import MasterWindow
                window: QMainWindow = MasterWindow()
            else:
                from slave.gui_slave import SlaveWindow
                window = SlaveWindow()
            windows.append(window)
            window.show()
            self.close()

    start = StartWindow()
    start.show()
    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
