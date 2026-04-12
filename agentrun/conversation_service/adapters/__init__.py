"""Conversation Service 框架适配器。

提供不同 Agent 开发框架的会话持久化适配器。
"""

from agentrun.conversation_service.adapters.langchain_adapter import (
    OTSChatMessageHistory,
)

# ADK adapter 依赖 google-adk，仅在安装了 google-adk 时可用
try:
    from agentrun.conversation_service.adapters.adk_adapter import (
        OTSSessionService,
    )
except ImportError:
    pass

# LangGraph adapter 依赖 langgraph，仅在安装了 langgraph 时可用
try:
    from agentrun.conversation_service.adapters.langgraph_adapter import (
        OTSCheckpointSaver,
    )
except ImportError:
    pass

__all__ = [
    "OTSChatMessageHistory",
    "OTSSessionService",
    "OTSCheckpointSaver",
]
