"""Tests for Feishu approval card patch updates (im.v1.message.patch).

Closes #8847 — approval card never updates because message.update (PUT)
silently ignores interactive cards. The fix adds patch_message() which uses
im.v1.message.patch, and send_exec_approval now includes update_multi: True
in the card config.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Feishu mock so FeishuAdapter can be imported without lark-oapi
# ---------------------------------------------------------------------------
def _ensure_feishu_mocks():
    """Provide stubs for lark-oapi / aiohttp.web so the import succeeds."""
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in (
            "lark_oapi", "lark_oapi.api.im.v1",
            "lark_oapi.event", "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
from gateway.platforms.feishu import FeishuAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> FeishuAdapter:
    """Create a FeishuAdapter with mocked internals."""
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    return adapter


# ===========================================================================
# patch_message — im.v1.message.patch for interactive cards
# ===========================================================================

class TestPatchMessage:
    """Test patch_message uses im.v1.message.patch API."""

    @pytest.mark.asyncio
    async def test_calls_patch_api(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_001"),
        )
        adapter._client.im.v1.message.patch = MagicMock(return_value=mock_response)

        card_json = json.dumps({"config": {}, "elements": []})
        result = await adapter.patch_message(
            message_id="msg_001",
            card_content=card_json,
        )

        assert result.success is True
        assert result.message_id == "msg_001"
        adapter._client.im.v1.message.patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None

        result = await adapter.patch_message(
            message_id="msg_001",
            card_content="{}",
        )
        assert result.success is False
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_patch_failure_returns_error(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: False,
            data=None,
            msg="card not found",
            code=220340,
        )
        adapter._client.im.v1.message.patch = MagicMock(return_value=mock_response)

        result = await adapter.patch_message(
            message_id="msg_missing",
            card_content="{}",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        adapter = _make_adapter()
        adapter._client.im.v1.message.patch = MagicMock(
            side_effect=RuntimeError("network error")
        )

        result = await adapter.patch_message(
            message_id="msg_001",
            card_content="{}",
        )
        assert result.success is False
        assert "network error" in result.error


# ===========================================================================
# _build_patch_message_body / _build_patch_message_request — builder helpers
# ===========================================================================

class TestPatchMessageBuilders:
    """Test patch message builder helpers."""

    def test_build_patch_message_body(self):
        body = FeishuAdapter._build_patch_message_body(content='{"key": "val"}')
        # When lark_oapi is mocked, it returns a MagicMock; just verify it doesn't crash
        assert body is not None

    def test_build_patch_message_request(self):
        body = FeishuAdapter._build_patch_message_body(content='{}')
        request = FeishuAdapter._build_patch_message_request(
            message_id="msg_test", request_body=body
        )
        assert request is not None


# ===========================================================================
# send_exec_approval — update_multi in card config
# ===========================================================================

class TestApprovalCardUpdateMulti:
    """Test that send_exec_approval includes update_multi: True in card config."""

    @pytest.mark.asyncio
    async def test_card_config_has_update_multi(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            await adapter.send_exec_approval(
                chat_id="oc_12345",
                command="rm -rf /",
                session_key="test-session",
            )

        card = json.loads(mock_send.call_args[1]["payload"])
        assert card["config"]["update_multi"] is True

    @pytest.mark.asyncio
    async def test_update_prompt_card_config_has_update_multi(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            await adapter.send_update_prompt(
                chat_id="oc_12345",
                prompt="Continue?",
                session_key="test-session",
            )

        card = json.loads(mock_send.call_args[1]["payload"])
        assert card["config"]["update_multi"] is True


# ===========================================================================
# _resolve_approval — now patches the card via API
# ===========================================================================

class TestResolveApprovalPatchesCard:
    """Test _resolve_approval patches the interactive card via patch_message."""

    @pytest.mark.asyncio
    async def test_patches_card_on_resolve(self):
        adapter = _make_adapter()
        adapter._approval_state[1] = {
            "session_key": "agent:main:feishu:group:oc_12345",
            "message_id": "msg_001",
            "chat_id": "oc_12345",
        }

        with (
            patch("tools.approval.resolve_gateway_approval", return_value=1),
            patch.object(adapter, "patch_message", new_callable=AsyncMock) as mock_patch,
        ):
            mock_patch.return_value = SimpleNamespace(success=True, message_id="msg_001")
            await adapter._resolve_approval(1, "once", "Alice")

        mock_patch.assert_called_once()
        call_kwargs = mock_patch.call_args[1]
        assert call_kwargs["message_id"] == "msg_001"
        # Verify patched card is the resolved approval card
        card = json.loads(call_kwargs["card_content"])
        assert card["header"]["template"] == "green"
        assert "Approved once" in card["header"]["title"]["content"]
        assert "Alice" in card["elements"][0]["content"]

    @pytest.mark.asyncio
    async def test_patches_card_on_deny(self):
        adapter = _make_adapter()
        adapter._approval_state[2] = {
            "session_key": "sess-2",
            "message_id": "msg_002",
            "chat_id": "oc_99",
        }

        with (
            patch("tools.approval.resolve_gateway_approval", return_value=1),
            patch.object(adapter, "patch_message", new_callable=AsyncMock) as mock_patch,
        ):
            mock_patch.return_value = SimpleNamespace(success=True, message_id="msg_002")
            await adapter._resolve_approval(2, "deny", "Bob")

        card = json.loads(mock_patch.call_args[1]["card_content"])
        assert card["header"]["template"] == "red"
        assert "Denied" in card["header"]["title"]["content"]

    @pytest.mark.asyncio
    async def test_no_patch_when_no_message_id(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = {
            "session_key": "sess-3",
            "message_id": "",
            "chat_id": "oc_99",
        }

        with (
            patch("tools.approval.resolve_gateway_approval", return_value=1),
            patch.object(adapter, "patch_message", new_callable=AsyncMock) as mock_patch,
        ):
            await adapter._resolve_approval(3, "once", "Charlie")

        mock_patch.assert_not_called()

    @pytest.mark.asyncio
    async def test_patch_failure_does_not_block_resolution(self):
        adapter = _make_adapter()
        adapter._approval_state[4] = {
            "session_key": "sess-4",
            "message_id": "msg_004",
            "chat_id": "oc_99",
        }

        with (
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
            patch.object(adapter, "patch_message", new_callable=AsyncMock) as mock_patch,
        ):
            mock_patch.return_value = SimpleNamespace(success=False, error="card not found")
            await adapter._resolve_approval(4, "once", "Dave")

        # Resolution should still happen even if patch fails
        mock_resolve.assert_called_once_with("sess-4", "once")
        assert 4 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_patch_exception_does_not_block_resolution(self):
        adapter = _make_adapter()
        adapter._approval_state[5] = {
            "session_key": "sess-5",
            "message_id": "msg_005",
            "chat_id": "oc_99",
        }

        with (
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
            patch.object(adapter, "patch_message", new_callable=AsyncMock) as mock_patch,
        ):
            mock_patch.side_effect = RuntimeError("timeout")
            await adapter._resolve_approval(5, "once", "Eve")

        # Resolution should still happen even if patch throws
        mock_resolve.assert_called_once_with("sess-5", "once")
        assert 5 not in adapter._approval_state
