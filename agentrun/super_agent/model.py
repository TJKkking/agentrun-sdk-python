"""Super Agent 数据模型 / Super Agent Data Models

此模块定义超级 Agent SDK 的输入输出数据模型。
This module defines the input/output data models for the Super Agent SDK.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SuperAgentCreateInput:
    """超级 Agent 创建输入 / Super Agent creation input"""

    name: str
    description: Optional[str] = None
    prompt: Optional[str] = None
    agents: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    sandboxes: List[str] = field(default_factory=list)
    workspaces: List[str] = field(default_factory=list)
    model_service_name: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class SuperAgentUpdateInput:
    """超级 Agent 更新输入 / Super Agent update input

    仅传想修改的字段, 其他保持不变。
    Only pass the fields to modify; others remain unchanged.
    """

    name: str
    description: Optional[str] = None
    prompt: Optional[str] = None
    agents: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    sandboxes: Optional[List[str]] = None
    workspaces: Optional[List[str]] = None
    model_service_name: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class SuperAgentListInput:
    """超级 Agent 列表查询输入 / Super Agent list query input"""

    page_number: int = 1
    page_size: int = 20


@dataclass
class InvokeResponseData:
    """Phase 1 响应 data 字段的强类型表示。

    Strongly-typed representation of the data field in the phase 1 response.
    """

    conversation_id: str
    stream_url: str
    stream_headers: Dict[str, str]


@dataclass
class Message:
    """会话消息 / Conversation message"""

    role: str
    content: str
    message_id: Optional[str] = None
    created_at: Optional[int] = None


@dataclass
class ConversationInfo:
    """服务端会话信息 / Server-side conversation info"""

    conversation_id: str
    agent_id: str
    title: Optional[str] = None
    main_user_id: Optional[str] = None
    sub_user_id: Optional[str] = None
    created_at: int = 0
    updated_at: int = 0
    error_message: Optional[str] = None
    invoke_info: Optional[Dict[str, Any]] = None
    messages: List[Message] = field(default_factory=list)
    params: Optional[Dict[str, Any]] = None
