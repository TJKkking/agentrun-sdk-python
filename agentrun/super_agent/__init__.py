"""Super Agent 模块 / Super Agent Module

独立的超级 Agent SDK, 面向写应用的开发者。用户无需感知底层 AgentRuntime 概念,
只需:

.. code-block:: python

    from agentrun.super_agent import SuperAgentClient

    client = SuperAgentClient()
    agent = await client.create_async(name="my-agent", prompt="你好")
    stream = await agent.invoke_async(messages=[{"role": "user", "content": "hi"}])
    print(stream.conversation_id)
    async for ev in stream:
        print(ev.event, ev.data)

可选的 AG-UI 强类型适配, 显式导入:

.. code-block:: python

    from agentrun.super_agent.agui import as_agui_events

    async for event in as_agui_events(stream):
        ...  # event 是 ``ag_ui.core.BaseEvent`` 子类实例

详见 ``openspec/changes/add-super-agent-sdk/`` 中的 proposal、design、spec。
"""

from .agent import SuperAgent
from .client import SuperAgentClient
from .model import (
    ConversationInfo,
    InvokeResponseData,
    Message,
    SuperAgentCreateInput,
    SuperAgentListInput,
    SuperAgentUpdateInput,
)
from .stream import InvokeStream, SSEEvent

__all__ = [
    "SuperAgentClient",
    "SuperAgent",
    "InvokeStream",
    "SSEEvent",
    "SuperAgentCreateInput",
    "SuperAgentUpdateInput",
    "SuperAgentListInput",
    "ConversationInfo",
    "Message",
    "InvokeResponseData",
]
