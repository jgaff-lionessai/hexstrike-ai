"""Regression checks for MCP startup and HTTP response handling.

Run with: python -m unittest discover -s tests -v
No security tools are executed; the integration check uses a local test server.
"""

import argparse
import asyncio
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import requests


SOURCE = Path(__file__).resolve().parents[1] / "hexstrike_mcp.py"
sys.path.insert(0, str(SOURCE.parent))
import hexstrike_mcp as client_module


class ClientTransportTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        session_factory = patch.object(client_module.requests, "Session", return_value=self.session)
        session_factory.start()
        self.addCleanup(session_factory.stop)
        logger = patch.object(client_module, "logger")
        logger.start()
        self.addCleanup(logger.stop)
        self.client = client_module.HexStrikeClient("http://127.0.0.1:8888/", timeout=17)
        self.addCleanup(self.client.close)

    def test_construction_does_not_contact_backend(self):
        self.assertEqual(self.session.mock_calls, [])

    def test_requests_use_expected_method_payload_and_timeout(self):
        response = self.session.request.return_value
        response.json.return_value = {"success": True}
        operations = [
            ("GET", lambda: self.client.safe_get("/health", {"detail": "tools"}), {"params": {"detail": "tools"}}),
            ("POST", lambda: self.client.safe_post("/health", {"detail": "tools"}), {"json": {"detail": "tools"}}),
            ("DELETE", lambda: self.client.safe_delete("/health", {"detail": "tools"}), {"json": {"detail": "tools"}}),
        ]
        for method, operation, payload in operations:
            with self.subTest(method=method):
                self.session.request.reset_mock()
                self.assertEqual(operation(), {"success": True})
                self.session.request.assert_called_once_with(
                    method, "http://127.0.0.1:8888/health", timeout=17, **payload
                )

    def operations(self):
        return [
            ("GET", lambda: self.client.safe_get("health")),
            ("POST", lambda: self.client.safe_post("health", {})),
            ("DELETE", lambda: self.client.safe_delete("health", {})),
        ]

    def assert_error(self, result):
        self.assertIsInstance(result, dict)
        self.assertIs(result.get("success"), False)
        self.assertTrue(result.get("error"))

    def test_non_object_json_is_a_structured_error(self):
        response = self.session.request.return_value
        for value in (None, [], "unexpected", True):
            response.json.return_value = value
            for method, operation in self.operations():
                with self.subTest(method=method, value=value):
                    self.assert_error(operation())

    def test_invalid_json_is_a_structured_error(self):
        self.session.request.return_value.json.side_effect = ValueError("invalid JSON")
        for method, operation in self.operations():
            with self.subTest(method=method):
                self.assert_error(operation())

    def test_request_failure_is_a_structured_error(self):
        self.session.request.side_effect = requests.exceptions.Timeout("backend did not respond")
        for method, operation in self.operations():
            with self.subTest(method=method):
                self.assert_error(operation())

    def test_http_error_is_a_structured_error(self):
        response = self.session.request.return_value
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("503 unavailable")
        for method, operation in self.operations():
            with self.subTest(method=method):
                self.assert_error(operation())
        response.json.assert_not_called()

    def test_client_rejects_nonpositive_timeout(self):
        for timeout in (0, -1):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                client_module.HexStrikeClient("http://127.0.0.1:8888", timeout=timeout)


class LoggingAndCliTests(unittest.TestCase):
    def test_formatter_preserves_shared_record(self):
        record = logging.LogRecord("test", logging.INFO, __file__, 0, "hello %s", ("world",), None)
        formatter = client_module.ColoredFormatter("%(message)s")
        first = formatter.format(record)
        self.assertEqual(formatter.format(record), first)
        self.assertEqual(record.msg, "hello %s")
        self.assertEqual(record.args, ("world",))
        self.assertEqual(logging.Formatter("%(message)s").format(record), "hello world")

    def test_import_leaves_host_logging_configuration_unchanged(self):
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(io.StringIO())
        formatter = logging.Formatter("host: %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        self.addCleanup(root_logger.removeHandler, handler)
        handlers_before = tuple(root_logger.handlers)
        formatters_before = tuple(item.formatter for item in handlers_before)
        level_before = root_logger.level
        spec = importlib.util.spec_from_file_location("hexstrike_logging_import_check", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(tuple(root_logger.handlers), handlers_before)
        self.assertEqual(tuple(item.formatter for item in root_logger.handlers), formatters_before)
        self.assertEqual(root_logger.level, level_before)

    def test_cli_rejects_invalid_timeouts(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as raised:
                    client_module.parse_args(["--timeout", value])
                self.assertEqual(raised.exception.code, 2)
        self.assertEqual(client_module.parse_args(["--timeout", "12"]).timeout, 12)

    def test_main_closes_session_after_server_exit(self):
        for failure in (None, RuntimeError("MCP failure"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                client = Mock()
                server = Mock()
                server.run.side_effect = failure
                args = argparse.Namespace(server="http://127.0.0.1:8888", timeout=17, debug=False)
                with patch.object(client_module, "parse_args", return_value=args), \
                     patch.object(client_module, "configure_logging"), \
                     patch.object(client_module, "logger"), \
                     patch.object(client_module, "HexStrikeClient", return_value=client), \
                     patch.object(client_module, "setup_mcp_server", return_value=server):
                    if isinstance(failure, RuntimeError):
                        with self.assertRaises(SystemExit) as raised:
                            client_module.main()
                        self.assertEqual(raised.exception.code, 1)
                    else:
                        client_module.main()
                client.close.assert_called_once_with()
                client.check_health.assert_not_called()

    def test_main_closes_session_when_server_setup_fails(self):
        client = Mock()
        args = argparse.Namespace(server="http://127.0.0.1:8888", timeout=17, debug=False)
        with patch.object(client_module, "parse_args", return_value=args), \
             patch.object(client_module, "configure_logging"), \
             patch.object(client_module, "logger"), \
             patch.object(client_module, "HexStrikeClient", return_value=client), \
             patch.object(client_module, "setup_mcp_server", side_effect=RuntimeError("setup failed")):
            with self.assertRaises(SystemExit) as raised:
                client_module.main()
            self.assertEqual(raised.exception.code, 1)
        client.close.assert_called_once_with()


class OfflineStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_does_not_wait_for_unresponsive_api(self):
        release_requests = threading.Event()

        class UnresponsiveHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.server.requests_seen += 1
                release_requests.wait(20)

            def log_message(self, format, *args):
                pass

        backend = ThreadingHTTPServer(("127.0.0.1", 0), UnresponsiveHandler)
        backend.requests_seen = 0
        backend_thread = threading.Thread(target=backend.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        backend_thread.start()
        url = f"http://127.0.0.1:{backend.server_port}"

        async def verify_startup():
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(SOURCE), "--server", url, "--timeout", "1"],
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
            )
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as error_log:
                async with stdio_client(params, errlog=error_log) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        self.assertIn("server_health", {tool.name for tool in tools.tools})
                        self.assertEqual(backend.requests_seen, 0)
                        response = await session.call_tool("server_health", {})
                        if response.structuredContent:
                            result = response.structuredContent.get("result", response.structuredContent)
                        else:
                            result = json.loads(next(item.text for item in response.content if item.type == "text"))
                        self.assertIs(result.get("success"), False)
                        self.assertTrue(result.get("error"))

        try:
            await asyncio.wait_for(verify_startup(), timeout=10)
        finally:
            release_requests.set()
            backend.shutdown()
            backend.server_close()
            backend_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
