"""Verify the super_agent module does not depend on conversation_service."""

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agentrun.super_agent as super_agent_pkg


def _python_files_under(path: Path):
    return [p for p in path.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_import_conversation_service():
    base = Path(super_agent_pkg.__file__).parent
    offenders = []
    for file in _python_files_under(base):
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("agentrun.conversation_service"):
                    offenders.append(str(file.relative_to(base)))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("agentrun.conversation_service"):
                        offenders.append(str(file.relative_to(base)))
    assert not offenders, (
        "super_agent must not import conversation_service, found in:"
        f" {offenders}"
    )


async def test_get_conversation_does_not_touch_session_store():
    """Calling agent.get_conversation_async must not touch SessionStore."""
    from agentrun.super_agent.agent import SuperAgent

    session_store_mock = MagicMock()
    # Patch in case code accidentally imported it
    with patch(
        "agentrun.conversation_service.SessionStore",
        session_store_mock,
        create=True,
    ):
        # Stub the data-plane call so we don't hit the network
        with patch(
            "agentrun.super_agent.agent.SuperAgentDataAPI"
        ) as data_api_factory:

            async def _noop(*args, **kwargs):
                return {}

            data_api_factory.return_value.get_conversation_async = _noop
            agent = SuperAgent(name="demo")
            await agent.get_conversation_async("c1")

    # SessionStore should never have been called/instantiated
    session_store_mock.assert_not_called()
