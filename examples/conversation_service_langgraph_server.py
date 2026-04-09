"""LangGraph Agent Server —— 使用 OTSCheckpointSaver 持久化 checkpoint。

集成步骤：
  Step 1: 初始化 SessionStore（OTS 后端）+ 创建 checkpoint 表
  Step 2: 创建 OTSCheckpointSaver
  Step 3: 构建 LangGraph StateGraph + 编译（传入 checkpointer）
  Step 4: 实现 invoke_agent，将 AgentRequest 转为 LangGraph 调用并流式输出
  Step 5: 通过 AgentRunServer 启动 HTTP 服务

使用方式：
  uv run --env-file .env python examples/conversation_service_langgraph_server.py

  # 请求示例（curl）：
    curl -X POST http://localhost:9001/openai/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "X-AgentRun-Session-ID: my-thread-2222" \
    -H "X-AgentRun-User-ID: user-1" \
    -d '{"model":"qwen3-max","stream":true,"messages":[{"role":"user","content":"你好"}]}'
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Any
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentrun import AgentRequest
from agentrun.conversation_service import SessionStore
from agentrun.conversation_service.adapters import OTSCheckpointSaver
from agentrun.server import AgentRunServer

load_dotenv()

# ── 配置参数 ──────────────────────────────────────────────────
AGENT_ID = "langgraph_chat_server"
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

if not MEMORY_COLLECTION_NAME:
    print("ERROR: 请设置环境变量 MEMORY_COLLECTION_NAME")
    sys.exit(1)
if not DASHSCOPE_API_KEY:
    print("ERROR: 请设置环境变量 DASHSCOPE_API_KEY")
    sys.exit(1)


# ── 定义 State ───────────────────────────────────────────────

from typing import TypedDict


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 定义 Graph 节点 ──────────────────────────────────────────

llm = ChatOpenAI(
    model="qwen-max",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def chat_node(state: ChatState) -> dict[str, Any]:
    """调用 LLM 生成回复。"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ── Step 1: 初始化 SessionStore + 创建 checkpoint 表 ─────────

store = SessionStore.from_memory_collection(MEMORY_COLLECTION_NAME)
store.init_langgraph_tables()

# ── Step 2: 创建 OTSCheckpointSaver ──────────────────────────

checkpointer = OTSCheckpointSaver(store, agent_id=AGENT_ID)

# ── Step 3: 构建 Graph ───────────────────────────────────────

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)
app = graph.compile(checkpointer=checkpointer)


# ── 辅助函数 ──────────────────────────────────────────────────


def _get_thread_id(req: AgentRequest) -> str:
    """从请求 header 中提取 thread_id（复用 session-id header）。"""
    raw_headers: dict[str, str] = {}
    if hasattr(req, "raw_request") and req.raw_request:
        raw_headers = dict(req.raw_request.headers)

    return (
        raw_headers.get("X-AgentRun-Session-ID")
        or raw_headers.get("x-agentrun-session-id")
        or raw_headers.get("X-Agentrun-Session-Id")
        or f"thread_{uuid.uuid4().hex[:8]}"
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


# ── Step 4: invoke_agent —— 核心 Server 处理函数 ─────────────


async def invoke_agent(req: AgentRequest):
    """将 AgentRequest 转换为 LangGraph 调用并流式输出。

    流程：
    1. 从 header 提取 thread_id / user_id
    2. 用 thread_id 作为 LangGraph 的 configurable.thread_id
    3. 提取最后一条用户消息
    4. 调用 graph.astream() 流式输出
    5. checkpoint 自动持久化到 OTS（由 OTSCheckpointSaver 处理）
    """
    thread_id = _get_thread_id(req)
    user_id = _get_user_id(req)
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"user_id": user_id},
    }

    # 展示当前 checkpoint 状态（体现 OTS 持久化能力）
    existing = await checkpointer.aget_tuple(config)
    if existing:
        msg_count = len(
            existing.checkpoint.get("channel_values", {}).get("messages", [])
        )
        print(
            f"[Thread {thread_id}] "
            f"user={user_id}, "
            f"已有 {msg_count} 条消息, "
            f"checkpoint_id={existing.checkpoint['id']}"
        )
    else:
        print(f"[Thread {thread_id}] user={user_id}, 新会话")

    # 提取最后一条用户消息
    last_user_text = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            last_user_text = msg.content or ""
            break

    if not last_user_text:
        yield "请输入您的问题。"
        return

    # 调用 LangGraph 流式输出
    try:
        async for event in app.astream(
            {"messages": [HumanMessage(content=last_user_text)]},
            config=config,
            stream_mode="values",
        ):
            messages = event.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    yield last_msg.content

        print(f"[Thread {thread_id}] 回复完成")

    except Exception as e:
        print(f"LangGraph 执行异常: {e}")
        raise Exception("Internal Error")


# ── Step 5: 启动 Server ──────────────────────────────────────

if __name__ == "__main__":
    server = AgentRunServer(
        invoke_agent=invoke_agent,
        memory_collection_name=MEMORY_COLLECTION_NAME,
    )
    print(f"Agent ID: {AGENT_ID}")
    print(f"Memory Collection: {MEMORY_COLLECTION_NAME}")
    print("请求时通过 X-AgentRun-Session-ID header 指定 thread_id")
    print("请求时通过 X-AgentRun-User-ID header 指定 user_id")
    server.start(port=9001)
