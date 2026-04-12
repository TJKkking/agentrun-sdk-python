"""LangChain Agent Server —— 使用 OTSChatMessageHistory 持久化消息历史。

集成步骤：
  Step 1: 初始化 SessionStore（OTS 后端）+ 创建 LangChain 所需表和索引
  Step 2: 构建 LangChain Chain（ChatOpenAI + SystemMessage）
  Step 3: 实现 invoke_agent，将 AgentRequest 转为 LangChain 调用并流式输出
  Step 4: 通过 AgentRunServer 启动 HTTP 服务

使用方式：
  uv run --env-file .env python examples/conversation_service_langchain_server.py

  # 请求示例（curl）：
    curl -X POST http://localhost:9002/openai/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "X-AgentRun-Session-ID: my-session-1" \
    -H "X-AgentRun-User-ID: user-1" \
    -d '{"model":"qwen-max","stream":true,"messages":[{"role":"user","content":"你好"}]}'
"""

from __future__ import annotations

import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agentrun import AgentRequest
from agentrun.conversation_service import SessionStore
from agentrun.conversation_service.adapters import OTSChatMessageHistory
from agentrun.server import AgentRunServer

load_dotenv()

# ── 配置参数 ──────────────────────────────────────────────────
AGENT_ID = "langchain_chat_server"
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

if not MEMORY_COLLECTION_NAME:
    print("ERROR: 请设置环境变量 MEMORY_COLLECTION_NAME")
    sys.exit(1)
if not DASHSCOPE_API_KEY:
    print("ERROR: 请设置环境变量 DASHSCOPE_API_KEY")
    sys.exit(1)


# ── Step 1: 初始化 SessionStore + 创建 LangChain 所需表和索引 ─

store = SessionStore.from_memory_collection(MEMORY_COLLECTION_NAME)
store.init_langchain_tables()

# ── Step 2: 构建 LangChain Chain ─────────────────────────────

SYSTEM_PROMPT = "你是一个友好的中文智能助手，请简洁、准确地回答用户问题。"

llm = ChatOpenAI(
    model="qwen-max",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=True,
)


# ── 辅助函数 ──────────────────────────────────────────────────


def _get_session_id(req: AgentRequest) -> str:
    """从请求 header 中提取 session_id，没有则生成一个。"""
    raw_headers: dict[str, str] = {}
    if hasattr(req, "raw_request") and req.raw_request:
        raw_headers = dict(req.raw_request.headers)

    return (
        raw_headers.get("X-AgentRun-Session-ID")
        or raw_headers.get("x-agentrun-session-id")
        or raw_headers.get("X-Agentrun-Session-Id")
        or f"session_{uuid.uuid4().hex[:8]}"
    )


def _get_user_id(req: AgentRequest) -> str:
    """从请求 header 中提取 user_id。"""
    raw_headers: dict[str, str] = {}
    if hasattr(req, "raw_request") and req.raw_request:
        raw_headers = dict(req.raw_request.headers)

    return (
        raw_headers.get("X-AgentRun-User-ID")
        or raw_headers.get("x-agentrun-user-id")
        or "default_user"
    )


# ── Step 3: invoke_agent —— 核心 Server 处理函数 ─────────────


async def invoke_agent(req: AgentRequest):
    """将 AgentRequest 转换为 LangChain 调用并流式输出。

    流程：
    1. 从 header 提取 session_id / user_id
    2. 创建 OTSChatMessageHistory（自动关联 / 创建 Session）
    3. 加载历史消息，展示 OTS 持久化状态
    4. 提取最后一条用户消息，写入历史
    5. 拼接 SystemMessage + 历史消息，调用 LLM 流式输出
    6. 将 AI 回复写入历史（自动持久化到 OTS）
    """
    session_id = _get_session_id(req)
    user_id = _get_user_id(req)

    # 创建消息历史（自动关联 / 创建 Session）
    history = OTSChatMessageHistory(
        session_store=store,
        agent_id=AGENT_ID,
        user_id=user_id,
        session_id=session_id,
    )

    # 展示当前持久化状态
    existing_messages = history.messages
    print(
        f"[Session {session_id}] "
        f"user={user_id}, "
        f"已有 {len(existing_messages)} 条消息"
    )

    # 提取最后一条用户消息
    last_user_text = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            last_user_text = msg.content or ""
            break

    if not last_user_text:
        yield "请输入您的问题。"
        return

    # 将用户消息写入历史
    history.add_message(HumanMessage(content=last_user_text))

    # 拼接完整消息列表：SystemMessage + 历史消息
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + history.messages

    # 调用 LLM 流式输出
    try:
        full_response = ""
        async for chunk in llm.astream(full_messages):
            text = chunk.content
            if isinstance(text, str) and text:
                full_response += text
                yield text

        # 将 AI 回复写入历史（持久化到 OTS）
        if full_response:
            history.add_message(AIMessage(content=full_response))

        print(
            f"[Session {session_id}] 回复完成，当前共"
            f" {len(history.messages)} 条消息"
        )

    except Exception as e:
        print(f"LangChain 执行异常: {e}")
        raise Exception("Internal Error")


# ── Step 4: 启动 Server ──────────────────────────────────────

if __name__ == "__main__":
    server = AgentRunServer(
        invoke_agent=invoke_agent,
        memory_collection_name=MEMORY_COLLECTION_NAME,
    )
    print(f"Agent ID: {AGENT_ID}")
    print(f"Memory Collection: {MEMORY_COLLECTION_NAME}")
    print("请求时通过 X-AgentRun-Session-ID header 指定会话 ID")
    print("请求时通过 X-AgentRun-User-ID header 指定 user_id")
    server.start(port=9002)
