"""Regression checks for MCP/API contracts without executing security tools."""

import ast
from collections import Counter
from pathlib import Path
import unittest

from hexstrike_mcp import setup_mcp_server


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self):
        self.calls = []

    def safe_get(self, endpoint, params=None):
        self.calls.append(("GET", endpoint, params))
        return {"success": True}

    def safe_post(self, endpoint, data):
        self.calls.append(("POST", endpoint, data))
        return {"success": True}

    def safe_delete(self, endpoint, data):
        self.calls.append(("DELETE", endpoint, data))
        return {"success": True}

    def check_health(self):
        return {"status": "healthy", "total_tools_available": 0}


class MCPToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = FakeClient()
        self.server = setup_mcp_server(self.client)

    async def test_delete_file_uses_delete_with_json_filename(self):
        await self.server.call_tool("delete_file", {"filename": "fixture.txt"})
        self.assertEqual(
            self.client.calls,
            [("DELETE", "api/files/delete", {"filename": "fixture.txt"})],
        )

    async def test_httpx_uses_the_supported_target_schema(self):
        tools = {tool.name: tool for tool in await self.server.list_tools()}
        schema = tools["httpx_probe"].inputSchema
        self.assertIn("target", schema["required"])
        self.assertNotIn("targets", schema["properties"])
        await self.server.call_tool(
            "httpx_probe", {"target": "fixture-targets.txt", "tech_detect": True}
        )
        method, endpoint, payload = self.client.calls[-1]
        self.assertEqual((method, endpoint), ("POST", "api/tools/httpx"))
        self.assertEqual(payload["target"], "fixture-targets.txt")
        self.assertTrue(payload["tech_detect"])

    async def test_nonexistent_burpsuite_endpoint_is_not_advertised(self):
        tools = {tool.name for tool in await self.server.list_tools()}
        self.assertNotIn("burpsuite_scan", tools)
        self.assertIn("burpsuite_alternative_scan", tools)

    async def test_paramspider_maps_output_file_to_api_output(self):
        await self.server.call_tool(
            "paramspider_discovery",
            {"domain": "example.test", "output_file": "fixture-output.txt"},
        )
        method, endpoint, payload = self.client.calls[-1]
        self.assertEqual((method, endpoint), ("POST", "api/tools/paramspider"))
        self.assertEqual(payload["output"], "fixture-output.txt")
        self.assertNotIn("output_file", payload)

    async def test_autorecon_defaults_match_the_api(self):
        await self.server.call_tool("autorecon_scan", {"target": "example.test"})
        method, endpoint, payload = self.client.calls[-1]
        self.assertEqual((method, endpoint), ("POST", "api/tools/autorecon"))
        self.assertEqual(
            payload,
            {
                "target": "example.test",
                "output_dir": "/tmp/autorecon",
                "heartbeat": 60,
                "timeout": 300,
                "port_scans": "top-100-ports",
                "service_scans": "default",
                "additional_args": "",
            },
        )
        tools = {tool.name: tool for tool in await self.server.list_tools()}
        self.assertEqual(
            set(tools["autorecon_scan"].inputSchema["properties"]), set(payload)
        )

    async def test_autorecon_preserves_supported_options(self):
        payload = {
            "target": "example.test",
            "output_dir": "/tmp/fixture-results",
            "heartbeat": 30,
            "timeout": 120,
            "port_scans": "default",
            "service_scans": "fixture-service",
            "additional_args": "",
        }
        await self.server.call_tool("autorecon_scan", payload)
        self.assertEqual(self.client.calls, [("POST", "api/tools/autorecon", payload)])

    async def test_arjun_sends_only_supported_api_fields(self):
        payload = {
            "url": "https://example.test",
            "method": "POST",
            "wordlist": "fixture-parameters.txt",
            "delay": 0.5,
            "threads": 10,
            "stable": True,
            "additional_args": "",
        }
        await self.server.call_tool("arjun_scan", payload)
        self.assertEqual(self.client.calls, [("POST", "api/tools/arjun", payload)])
        tools = {tool.name: tool for tool in await self.server.list_tools()}
        self.assertEqual(set(tools["arjun_scan"].inputSchema["properties"]), set(payload))


class APIContractTests(unittest.TestCase):
    def test_tool_definition_names_are_unique(self):
        module = ast.parse((ROOT / "hexstrike_mcp.py").read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "tool"
                for decorator in node.decorator_list
            ):
                names.append(node.name)
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        self.assertEqual(duplicates, [])

    def test_constant_api_calls_match_server_routes_and_methods(self):
        # Inspect the backend source without importing or starting its tool engines.
        backend = ast.parse((ROOT / "hexstrike_server.py").read_text(encoding="utf-8"))
        routes = {}
        for node in ast.walk(backend):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "route"
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    continue
                methods = {"GET"}
                for keyword in decorator.keywords:
                    if keyword.arg == "methods":
                        methods = set(ast.literal_eval(keyword.value))
                routes[decorator.args[0].value.lstrip("/")] = methods

        client = ast.parse((ROOT / "hexstrike_mcp.py").read_text(encoding="utf-8"))
        checked = 0
        for call in ast.walk(client):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"safe_get", "safe_post", "safe_delete"}
                and call.args
                and isinstance(call.args[0], ast.Constant)
            ):
                continue
            endpoint = call.args[0].value
            method = call.func.attr.removeprefix("safe_").upper()
            with self.subTest(endpoint=endpoint, method=method, line=call.lineno):
                self.assertIn(endpoint, routes)
                self.assertIn(method, routes[endpoint])
            checked += 1
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()
