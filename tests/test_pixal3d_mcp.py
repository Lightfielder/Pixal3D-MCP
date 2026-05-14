#!/usr/bin/env python
"""
Unit tests for Pixal3D MCP Server

These tests validate the MCP server functionality including:
- JSON-RPC 2.0 protocol compliance
- Tool definitions and handlers
- Conda environment functions
- Memory management functions

Run with:
    cd Pixal3D-MCP
    python3 -m pytest tests/ -v
    python3 -m unittest tests/test_pixal3d_mcp.py -v
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Path configuration - resolve relative to this test file's location
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)  # tests/ -> Pixal3D-MCP/
PIXAL3D_MCP_SERVER_PATH = os.path.join(PROJECT_ROOT, "src", "pixal3d_mcp", "server.py")

# Add src to path
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Import the MCP server module
import importlib.util

spec = importlib.util.spec_from_file_location(
    "pixal3d_mcp_server", PIXAL3D_MCP_SERVER_PATH
)
pixal3d_mcp_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pixal3d_mcp_server)


class TestModuleLoading(unittest.TestCase):
    """Test that Pixal3D MCP module loads correctly"""

    def test_server_module_imports(self):
        """Server module should be importable"""
        self.assertIsNotNone(pixal3d_mcp_server)

    def test_global_state_initialized(self):
        """Global state should be initialized (pipeline not loaded yet)"""
        self.assertIsNone(pixal3d_mcp_server.PIPELINE)  # Pipeline starts as None
        self.assertFalse(pixal3d_mcp_server.PIPELINE_LOADED)  # Not loaded yet


class TestJSONRPCProtocol(unittest.TestCase):
    """Test JSON-RPC 2.0 protocol compliance"""

    def test_initialize_response_has_jsonrpc_field(self):
        """Initialize response must have jsonrpc: '2.0'"""
        request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("jsonrpc", response)
        self.assertEqual(response["jsonrpc"], "2.0")

    def test_initialize_response_has_protocol_version(self):
        """Initialize response must have protocolVersion"""
        request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("result", response)
        self.assertIn("protocolVersion", response["result"])

    def test_initialize_response_has_server_info(self):
        """Initialize response must have serverInfo with name and version"""
        request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("result", response)
        self.assertIn("serverInfo", response["result"])
        self.assertIn("name", response["result"]["serverInfo"])
        self.assertIn("version", response["result"]["serverInfo"])
        self.assertEqual(
            response["result"]["serverInfo"]["name"], "Pixal3D Image-to-3D MCP Server"
        )

    def test_initialize_response_has_capabilities(self):
        """Initialize response must have capabilities"""
        request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("result", response)
        self.assertIn("capabilities", response["result"])

    def test_tools_list_response_has_jsonrpc_field(self):
        """Tools/list response must have jsonrpc: '2.0'"""
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("jsonrpc", response)
        self.assertEqual(response["jsonrpc"], "2.0")

    def test_tools_list_response_has_tools(self):
        """Tools/list response must have tools array"""
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("result", response)
        self.assertIn("tools", response["result"])
        self.assertIsInstance(response["result"]["tools"], list)

    def test_tools_call_response_has_jsonrpc_field(self):
        """Tools/call response must have jsonrpc: '2.0'"""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_conda_envs", "arguments": {}},
        }
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("jsonrpc", response)
        self.assertEqual(response["jsonrpc"], "2.0")

    def test_shutdown_response_has_jsonrpc_field(self):
        """Shutdown response must have jsonrpc: '2.0'"""
        request = {"jsonrpc": "2.0", "id": 4, "method": "shutdown", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("jsonrpc", response)
        self.assertEqual(response["jsonrpc"], "2.0")

    def test_unknown_method_returns_error_with_jsonrpc(self):
        """Unknown method should return error with jsonrpc: '2.0'"""
        request = {"jsonrpc": "2.0", "id": 5, "method": "unknown/method", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("jsonrpc", response)
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertIn("error", response)

    def test_unknown_tool_returns_error(self):
        """Unknown tool should return error with code -32601"""
        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
        response = pixal3d_mcp_server.handle_request(request)
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32601)


class TestToolDefinitions(unittest.TestCase):
    """Test that tool definitions are properly defined"""

    def test_tools_defined(self):
        """Tool definitions should be loaded"""
        self.assertGreater(len(pixal3d_mcp_server.TOOL_DEFINITIONS), 0)

    def test_tool_has_name(self):
        """Each tool should have a name"""
        for tool in pixal3d_mcp_server.TOOL_DEFINITIONS:
            self.assertIn("name", tool)
            self.assertIsInstance(tool["name"], str)
            self.assertGreater(len(tool["name"]), 0)

    def test_tool_has_description(self):
        """Each tool should have a description"""
        for tool in pixal3d_mcp_server.TOOL_DEFINITIONS:
            self.assertIn("description", tool)
            self.assertIsInstance(tool["description"], str)

    def test_tool_has_input_schema(self):
        """Each tool should have inputSchema"""
        for tool in pixal3d_mcp_server.TOOL_DEFINITIONS:
            self.assertIn("inputSchema", tool)
            self.assertIsInstance(tool["inputSchema"], dict)
            self.assertIn("type", tool["inputSchema"])
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertIn("properties", tool["inputSchema"])

    def test_all_tools_have_unique_names(self):
        """All tools should have unique names"""
        names = [tool["name"] for tool in pixal3d_mcp_server.TOOL_DEFINITIONS]
        self.assertEqual(len(names), len(set(names)), "Tool names must be unique")


class TestToolHandlers(unittest.TestCase):
    """Test that all tools have corresponding handlers"""

    def test_load_pipeline_handler_exists(self):
        """load_pipeline handler should exist"""
        self.assertIn("load_pipeline", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_unload_pipeline_handler_exists(self):
        """unload_pipeline handler should exist"""
        self.assertIn("unload_pipeline", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_run_inference_handler_exists(self):
        """run_inference handler should exist"""
        self.assertIn("run_inference", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_get_memory_status_handler_exists(self):
        """get_memory_status handler should exist"""
        self.assertIn("get_memory_status", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_cleanup_memory_handler_exists(self):
        """cleanup_memory handler should exist"""
        self.assertIn("cleanup_memory", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_load_conda_env_handler_exists(self):
        """load_conda_env handler should exist"""
        self.assertIn("load_conda_env", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_list_conda_envs_handler_exists(self):
        """list_conda_envs handler should exist"""
        self.assertIn("list_conda_envs", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_get_conda_env_status_handler_exists(self):
        """get_conda_env_status handler should exist"""
        self.assertIn("get_conda_env_status", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_handlers_and_tools_match(self):
        """All handlers should have corresponding tools"""
        handler_names = set(pixal3d_mcp_server.TOOL_HANDLERS.keys())
        tool_names = set(tool["name"] for tool in pixal3d_mcp_server.TOOL_DEFINITIONS)
        self.assertEqual(handler_names, tool_names, "Handlers and tools must match")


class TestCondaEnvironmentFunctions(unittest.TestCase):
    """Test conda environment functions"""

    def test_list_conda_envs_returns_structure(self):
        """list_conda_envs should return proper structure"""
        # This will fail if conda is not available, which is expected in CI
        result = pixal3d_mcp_server.list_conda_envs()
        self.assertIn("status", result)
        # Result should be either success or error, never invalid
        self.assertIn(result["status"], ["success", "error"])

    @patch("subprocess.run")
    def test_list_conda_envs_parses_output(self, mock_run):
        """list_conda_envs should parse conda env list output"""
        mock_run.return_value = MagicMock(
            stdout="# conda environments:\nbase                  /home/user/miniconda3\npixal3d             /home/user/miniconda3/envs/pixal3d\n",
            returncode=0,
        )
        result = pixal3d_mcp_server.list_conda_envs()
        self.assertEqual(result["status"], "success")
        self.assertIn("environments", result)
        self.assertIsInstance(result["environments"], list)

    @patch("subprocess.run")
    def test_load_conda_env_returns_error_for_missing_env(self, mock_run):
        """load_conda_env should return error for non-existent environment"""
        mock_run.return_value = MagicMock(
            stdout="# conda environments:\nbase                  /home/user/miniconda3\n",
            returncode=0,
        )
        result = pixal3d_mcp_server.load_conda_env("nonexistent_env_xyz")
        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)

    def test_get_conda_env_status_returns_structure(self):
        """get_conda_env_status should return proper structure"""
        result = pixal3d_mcp_server.get_conda_env_status()
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        self.assertIn("current_env", result)
        self.assertIn("conda_prefix", result)
        self.assertIn("env_name", result)
        self.assertIn("python_path", result)


class TestMemoryFunctions(unittest.TestCase):
    """Test memory management functions"""

    def test_get_memory_status_returns_structure(self):
        """get_memory_status should return proper structure"""
        result = pixal3d_mcp_server.handle_get_memory_status({})
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        # Should indicate if pipeline is loaded
        self.assertIn("pipeline_loaded", result)

    def test_cleanup_memory_returns_success(self):
        """cleanup_memory should return success status"""
        result = pixal3d_mcp_server.handle_cleanup_memory({})
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")

    def test_handle_load_conda_env_returns_structure(self):
        """handle_load_conda_env should return proper structure"""
        result = pixal3d_mcp_server.handle_load_conda_env(
            {"env_name": "nonexistent_xyz"}
        )
        self.assertIn("status", result)
        # Either success or error, never invalid
        self.assertIn(result["status"], ["success", "error"])

    def test_handle_get_conda_env_status_returns_structure(self):
        """handle_get_conda_env_status should return proper structure"""
        result = pixal3d_mcp_server.handle_get_conda_env_status({})
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")

    def test_handle_list_conda_envs_returns_structure(self):
        """handle_list_conda_envs should return proper structure"""
        result = pixal3d_mcp_server.handle_list_conda_envs({})
        self.assertIn("status", result)
        self.assertIn(result["status"], ["success", "error"])


class TestRunInferenceValidation(unittest.TestCase):
    """Test run_inference parameter validation"""

    def test_run_inference_requires_image(self):
        """run_inference should require image parameter"""
        result = pixal3d_mcp_server.handle_run_inference({})
        self.assertEqual(result["status"], "error")
        self.assertIn("image", result["message"].lower())

    def test_run_inference_accepts_valid_params(self):
        """run_inference should accept valid parameters"""
        # This will fail at execution time (no actual image file)
        # but validation should pass
        result = pixal3d_mcp_server.handle_run_inference(
            {
                "image": "test.png",
                "output": "output.glb",
                "seed": 42,
                "max_num_tokens": 4096,
            }
        )
        # Should be error from execution, not validation
        self.assertEqual(result["status"], "error")
        self.assertNotIn("image", result.get("message", "").lower())

    def test_run_inference_default_values(self):
        """run_inference should use default values for optional params"""
        result = pixal3d_mcp_server.handle_run_inference({"image": "test.png"})
        # Should not fail on validation, only on execution
        self.assertEqual(result["status"], "error")
        # Error should be about file not found or CUDA, not missing params
        error_msg = result.get("message", "").lower()
        self.assertNotIn("required", error_msg)


class TestLoadUnloadPipeline(unittest.TestCase):
    """Test pipeline load/unload functions"""

    def test_unload_pipeline_returns_structure(self):
        """unload_pipeline should return proper structure"""
        result = pixal3d_mcp_server.handle_unload_pipeline({})
        self.assertIn("status", result)

    def test_handle_load_pipeline_returns_structure(self):
        """handle_load_pipeline should return proper structure"""
        result = pixal3d_mcp_server.handle_load_pipeline({})
        self.assertIn("status", result)
        self.assertIn(result["status"], ["success", "error"])


class TestEndToEndMCPFlow(unittest.TestCase):
    """Test complete MCP request/response flows"""

    def test_initialize_flow(self):
        """Test complete initialize flow"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        }
        response = pixal3d_mcp_server.handle_request(request)

        self.assertEqual(response["id"], 1)
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertIn("result", response)
        self.assertEqual(
            response["result"]["serverInfo"]["name"], "Pixal3D Image-to-3D MCP Server"
        )

    def test_tools_list_flow(self):
        """Test complete tools/list flow"""
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)

        self.assertEqual(response["id"], 2)
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertIn("result", response)

        tools = response["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        # Verify all expected tools are present
        expected_tools = [
            "load_pipeline",
            "unload_pipeline",
            "run_inference",
            "run_inference_sequence",
            "preview_images",
            "preview_model",
            "preview_model_sequence",
            "get_memory_status",
            "cleanup_memory",
            "load_conda_env",
            "list_conda_envs",
            "get_conda_env_status",
        ]
        for tool in expected_tools:
            self.assertIn(tool, tool_names, f"Missing tool: {tool}")

    def test_tool_call_flow(self):
        """Test complete tool call flow"""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_conda_env_status", "arguments": {}},
        }
        response = pixal3d_mcp_server.handle_request(request)

        self.assertEqual(response["id"], 3)
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertIn("result", response)
        self.assertFalse(response["result"].get("isError", False))
        self.assertIn("content", response["result"])

    def test_shutdown_flow(self):
        """Test complete shutdown flow"""
        request = {"jsonrpc": "2.0", "id": 4, "method": "shutdown", "params": {}}
        response = pixal3d_mcp_server.handle_request(request)

        self.assertEqual(response["id"], 4)
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertIn("result", response)


class TestPreviewImagesTool(unittest.TestCase):
    """Test preview_images tool functionality"""

    def test_preview_images_handler_exists(self):
        """preview_images handler should exist"""
        self.assertIn("preview_images", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_handle_preview_images_requires_input(self):
        """preview_images should require either image_paths or input_path"""
        result = pixal3d_mcp_server.handle_preview_images({})
        self.assertEqual(result["status"], "error")
        self.assertIn("required", result["message"].lower())

    @patch("os.path.exists")
    def test_preview_images_with_single_file(self, mock_exists):
        """preview_images should handle single file with mock"""
        mock_exists.return_value = True
        with patch.object(pixal3d_mcp_server, "generate_thumbnail") as mock_thumb:
            mock_thumb.return_value = {
                "thumbnail": "base64data",
                "width": 1920,
                "height": 1080,
                "thumbnail_width": 256,
                "thumbnail_height": 144,
            }
            result = pixal3d_mcp_server.preview_images(image_paths=["test.png"])
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["count"], 1)

    def test_preview_images_returns_structure(self):
        """preview_images should return proper structure"""
        with patch.object(pixal3d_mcp_server, "generate_thumbnail") as mock_thumb:
            mock_thumb.return_value = {"error": "File not found"}
            result = pixal3d_mcp_server.preview_images(image_paths=["nonexistent.png"])
            self.assertIn("status", result)
            self.assertIn(result["status"], ["success", "error"])


class TestPreviewModelTool(unittest.TestCase):
    """Test preview_model tool functionality"""

    def test_preview_model_handler_exists(self):
        """preview_model handler should exist"""
        self.assertIn("preview_model", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_handle_preview_model_requires_input(self):
        """preview_model should require either glb_paths or input_path"""
        result = pixal3d_mcp_server.handle_preview_model({})
        self.assertEqual(result["status"], "error")
        self.assertIn("required", result["message"].lower())

    def test_preview_model_returns_error_for_no_glb_files(self):
        """preview_model should return error when no GLB files found"""
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = []
            result = pixal3d_mcp_server.preview_model(input_path="/*.glb")
            self.assertEqual(result["status"], "error")
            self.assertIn("no glb", result["message"].lower())

    @patch.object(pixal3d_mcp_server, "render_glb_preview")
    def test_preview_model_with_mock_rendering(self, mock_render):
        """preview_model should process GLB files with mock rendering"""
        mock_render.return_value = {
            "preview": "base64pngdata",
            "width": 512,
            "height": 512,
        }
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["test.glb"]
            result = pixal3d_mcp_server.preview_model(input_path="test.glb")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["count"], 1)


class TestPreviewModelSequenceTool(unittest.TestCase):
    """Test preview_model_sequence tool functionality"""

    def test_preview_model_sequence_handler_exists(self):
        """preview_model_sequence handler should exist"""
        self.assertIn("preview_model_sequence", pixal3d_mcp_server.TOOL_HANDLERS)

    def test_handle_preview_model_sequence_requires_input(self):
        """preview_model_sequence should require input_path"""
        result = pixal3d_mcp_server.handle_preview_model_sequence({})
        self.assertEqual(result["status"], "error")
        self.assertIn("required", result["message"].lower())

    @patch.object(pixal3d_mcp_server, "preview_model")
    def test_preview_model_sequence_with_frame_range(self, mock_preview):
        """preview_model_sequence should pass through to preview_model"""
        mock_preview.return_value = {
            "status": "success",
            "count": 2,
            "previews": [
                {"filename": "mesh_0001.glb"},
                {"filename": "mesh_0002.glb"},
            ],
        }
        result = pixal3d_mcp_server.handle_preview_model_sequence(
            {"input_path": "mesh_####.glb", "frame_start": 1, "frame_end": 10}
        )
        self.assertEqual(result["status"], "success")


class TestRunInferenceSequenceValidation(unittest.TestCase):
    """Test run_inference_sequence parameter handling"""

    def test_handle_run_inference_sequence_requires_input(self):
        """run_inference_sequence should require input_path"""
        result = pixal3d_mcp_server.handle_run_inference_sequence({})
        self.assertEqual(result["status"], "error")
        self.assertIn("input_path", result["message"].lower())

    def test_run_inference_sequence_handler_exists(self):
        """run_inference_sequence handler should exist"""
        self.assertIn("run_inference_sequence", pixal3d_mcp_server.TOOL_HANDLERS)


class TestThumbnailGeneration(unittest.TestCase):
    """Test thumbnail generation functionality"""

    def test_generate_thumbnail_returns_error_for_missing_file(self):
        """generate_thumbnail should return error for missing file"""
        result = pixal3d_mcp_server.generate_thumbnail("nonexistent_file.png")
        self.assertIn("error", result)


def run_tests():
    """Run all tests and print results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModuleLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestJSONRPCProtocol))
    suite.addTests(loader.loadTestsFromTestCase(TestToolDefinitions))
    suite.addTests(loader.loadTestsFromTestCase(TestToolHandlers))
    suite.addTests(loader.loadTestsFromTestCase(TestCondaEnvironmentFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestRunInferenceValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadUnloadPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndMCPFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestPreviewImagesTool))
    suite.addTests(loader.loadTestsFromTestCase(TestPreviewModelTool))
    suite.addTests(loader.loadTestsFromTestCase(TestPreviewModelSequenceTool))
    suite.addTests(loader.loadTestsFromTestCase(TestRunInferenceSequenceValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestThumbnailGeneration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("PIXAL3D-MCP TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print("\nSOME TESTS FAILED!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
