"""Проверка протокола «4 байта длины + JSON»."""

from __future__ import annotations

import socket
import struct
import threading
import unittest

from common.protocol import (ProtocolError, encode_message, recv_exact, recv_message,
                             send_message)


class TestProtocol(unittest.TestCase):
    def setUp(self) -> None:
        self.a, self.b = socket.socketpair()
        self.a.settimeout(5)
        self.b.settimeout(5)

    def tearDown(self) -> None:
        self.a.close()
        self.b.close()

    def test_roundtrip(self) -> None:
        msg = {"type": "RESULT", "partial_sum": 1.2345678901234567, "name": "узел-1"}
        send_message(self.a, msg)
        self.assertEqual(recv_message(self.b), msg)

    def test_frame_format(self) -> None:
        data = encode_message({"type": "HEARTBEAT"})
        (length,) = struct.unpack(">I", data[:4])
        self.assertEqual(length, len(data) - 4)

    def test_several_messages_in_a_row(self) -> None:
        for i in range(50):
            send_message(self.a, {"type": "PROGRESS", "progress": i / 50})
        for i in range(50):
            self.assertEqual(recv_message(self.b)["progress"], i / 50)

    def test_large_message(self) -> None:
        msg = {"type": "TASK", "data": "x" * 2_000_000}
        t = threading.Thread(target=send_message, args=(self.a, msg))
        t.start()
        self.assertEqual(recv_message(self.b), msg)
        t.join()

    def test_partial_chunks(self) -> None:
        """Сообщение приходит по одному байту: recv_exact должен собрать его целиком."""
        data = encode_message({"type": "REGISTER", "name": "slave", "cores": 8})

        def dribble() -> None:
            for byte in data:
                self.a.sendall(bytes([byte]))

        t = threading.Thread(target=dribble)
        t.start()
        self.assertEqual(recv_message(self.b), {"type": "REGISTER", "name": "slave", "cores": 8})
        t.join()

    def test_eof(self) -> None:
        self.a.close()
        self.assertIsNone(recv_message(self.b))

    def test_broken_in_the_middle(self) -> None:
        data = encode_message({"type": "RESULT", "partial_sum": 1.0})
        self.a.sendall(data[:-3])
        self.a.close()
        with self.assertRaises(ConnectionError):
            recv_message(self.b)

    def test_recv_exact(self) -> None:
        self.a.sendall(b"abc")
        self.a.sendall(b"def")
        self.assertEqual(recv_exact(self.b, 6), b"abcdef")

    def test_bad_json(self) -> None:
        self.a.sendall(struct.pack(">I", 3) + b"{{{")
        with self.assertRaises(ProtocolError):
            recv_message(self.b)

    def test_missing_type(self) -> None:
        payload = b'{"x": 1}'
        self.a.sendall(struct.pack(">I", len(payload)) + payload)
        with self.assertRaises(ProtocolError):
            recv_message(self.b)


if __name__ == "__main__":
    unittest.main()
