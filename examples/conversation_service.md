**Agent 的本质是对无状态 LLM 进行有状态的精细化 Context 管理**。会话（Session）与状态（State）是 LLM Context 的核心来源。因此，构建一个健壮的会话管理系统，不仅能显著提升开发者的体验，更是 Agent 运行平台的核心竞争力。

本文将介绍使用不同 Agent 开发框架如何接入、使用 AgentRun 提供的会话状态持久化能力。

## 1. 概述
AgentRun 提供了**会话状态持久化服务**，为 AI Agent 应用提供开箱即用的会话管理能力。它将会话元数据、对话事件流和多级状态统一持久化到阿里云 TableStore（OTS），让 Agent 具备**跨请求、跨重启**的记忆能力。

### 核心能力
| 能力 | 说明 |
| --- | --- |
| 会话管理 | 创建、查询、列出、删除会话，支持按时间排序和多元索引搜索 |
| 事件流持久化 | 自动持久化对话中的每一轮交互（用户消息、Agent 回复、工具调用等） |
| 三级状态管理 | 支持 App 级、User 级、Session 级三层状态，自动合并返回 |
| 多元索引搜索 | 按关键词、标签、时间范围等条件搜索会话 |
| 多框架适配 | 通过薄适配层对接不同 Agent 开发框架，应用代码无需感知底层存储 |


### 框架支持状态
| 框架 | 适配器 | 状态 |
| --- | --- | --- |
| Google ADK | `OTSSessionService` | 已支持 |
| LangChain | `OTSChatMessageHistory` | 即将推出 |
| LangGraph | - | 即将推出 |


---

## 2. 前置条件
### 2.1 创建 MemoryCollection
在 AgentRun 平台上创建一个 MemoryCollection 资源。MemoryCollection 内部包含了 TableStore 实例的连接信息（endpoint、instance_name），Conversation Service 会自动从中读取这些配置。

创建方式请参考 [AgentRun 官方文档](https://help.aliyun.com/zh/functioncompute/fc/memory-storage?spm=a2c4g.11186623.help-menu-2508973.d_3_11.3e076abaLO38Z2)。

### 2.2 配置环境变量
在运行应用前，请设置以下环境变量：

```bash
# 必填：MemoryCollection 名称（在 AgentRun 平台上创建的资源名称）
export MEMORY_COLLECTION_NAME="your-memory-collection-name"
```

也可以使用 `.env` 文件配合 `python-dotenv` 加载：

```plain
MEMORY_COLLECTION_NAME=your-memory-collection-name
```

> Conversation Service 也支持备选环境变量 `ALIBABA_CLOUD_ACCESS_KEY_ID` / `ALIBABA_CLOUD_ACCESS_KEY_SECRET`，SDK 会按优先级自动查找。
>

### 2.4 Python 环境
+ Python 3.10 及以上版本

---

## 3. 安装
```bash
pip install agentrun-sdk
```

如果需要使用 Google ADK 集成，还需安装 ADK 及模型调用依赖：

```bash
pip install google-adk litellm
```

---

## 4. 快速开始（Google ADK）
以下是一个最小可运行的示例，展示如何用 5 步将 Google ADK Agent 的会话持久化到 OTS。

> 示例中使用 DashScope 的 OpenAI 兼容接口，需要设置环境变量 `DASHSCOPE_API_KEY`。
>

```python
import asyncio
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.genai import types

from agentrun.conversation_service import SessionStore
from agentrun.conversation_service.adapters import OTSSessionService

# ── Step 1: 初始化 SessionStore ──────────────────────────────
store = SessionStore.from_memory_collection(
    os.environ["MEMORY_COLLECTION_NAME"]
)
store.init_tables()

# ── Step 2: 创建 OTSSessionService ──────────────────────────
session_service = OTSSessionService(session_store=store)

# ── Step 3: 创建 Agent + Runner ──────────────────────────────
model = LiteLlm(
    model="openai/qwen3-max",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
agent = Agent(
    name="assistant",
    model=model,
    instruction="你是一个友好的中文智能助手。",
)
runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
)


# ── Step 4: 对话（自动持久化到 OTS） ────────────────────────
async def main():
    # 创建会话
    session = await session_service.create_session(
        app_name="my_app",
        user_id="user_1",
    )
    print(f"会话已创建: {session.id}")

    # 发送消息
    content = types.Content(
        role="user",
        parts=[types.Part(text="你好，介绍一下你自己")],
    )
    async for event in runner.run_async(
        user_id="user_1",
        session_id=session.id,
        new_message=content,
    ):
        if (
            event.is_final_response()
            and event.content
            and event.content.parts
        ):
            for part in event.content.parts:
                if part.text:
                    print(f"Agent: {part.text}")

    # ── Step 5: 验证持久化 ───────────────────────────────────
    # 重新从 OTS 加载会话，确认事件已持久化
    loaded = await session_service.get_session(
        app_name="my_app",
        user_id="user_1",
        session_id=session.id,
    )
    print(f"持久化事件数: {len(loaded.events)}")


asyncio.run(main())
```

运行后，您会看到 Agent 的回复以及持久化的事件数量。即使程序重启，再次通过 `get_session` 加载同一个 `session_id`，历史对话仍然存在。

---

## 5. Google ADK 详细指南
### 5.1 初始化 SessionStore
SessionStore 是 Conversation Service 的核心入口。通过 MemoryCollection 名称初始化，SDK 会自动完成以下工作：

1. 调用 AgentRun API 获取 MemoryCollection 配置
2. 从中提取 TableStore 的 endpoint 和 instance_name
3. 构建 OTS 客户端

```python
from agentrun.conversation_service import SessionStore

store = SessionStore.from_memory_collection("your-memory-collection-name")
```

初始化后，需要调用 `init_tables()` 创建所需的数据库表和索引。该方法是**幂等**的，表或索引已存在时会自动跳过，不会报错，可以安全地在每次启动时调用。

```python
store.init_tables()
```

`init_tables()` 会创建以下资源：

| 资源 | 说明 |
| --- | --- |
| `conversation` 表 | 存储会话元信息（摘要、标签、时间戳等） |
| `event` 表 | 存储对话事件流（消息、工具调用等） |
| `state` 表 | 存储 Session 级状态 |
| `app_state` 表 | 存储 App 级状态 |
| `user_state` 表 | 存储 User 级状态 |
| `conversation_secondary_index` | 二级索引，支持按更新时间排序列出会话 |
| `conversation_search_index` | 多元索引，支持全文搜索和组合过滤 |
| `state_search_index` | 多元索引，支持按 session_id 独立查询状态 |


> **按需建表**：如果您的场景不需要全部功能，也可以分步创建：
>

| 方法 | 创建的资源 | 适用场景 |
| --- | --- | --- |
| `init_core_tables()` | conversation + event + 二级索引 | 仅需会话和事件，无三级 State |
| `init_state_tables()` | state + app_state + user_state | 仅补建 State 表 |
| `init_search_index()` | conversation + state 多元索引 | 仅补建搜索索引 |
| `init_tables()` | 以上全部 | 推荐，一次创建所有资源 |


#### 异步初始化
SessionStore 的所有方法均提供异步版本（方法名加 `_async` 后缀）：

```python
store = await SessionStore.from_memory_collection_async(
    "your-memory-collection-name"
)
await store.init_tables_async()
```

#### 表名前缀
如果多个应用共用同一个 OTS 实例，可以通过 `table_prefix` 参数隔离表名：

```python
store = SessionStore.from_memory_collection(
    "your-memory-collection-name",
    table_prefix="myapp_",
)
# 创建的表名为: myapp_conversation, myapp_event, myapp_state, ...
```

### 5.2 创建 OTSSessionService
`OTSSessionService` 是 Google ADK `BaseSessionService` 的 OTS 实现。将它传给 ADK 的 `Runner`，即可让 ADK 的会话自动持久化到 OTS。

```python
from agentrun.conversation_service.adapters import OTSSessionService

session_service = OTSSessionService(session_store=store)
```

然后将 `session_service` 传给 `Runner`：

```python
from google.adk.runners import Runner

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
)
```

此后，通过 `runner.run_async()` 进行的所有对话都会自动持久化到 OTS，包括：

+ 用户消息
+ Agent 回复
+ 工具调用（function_call）和工具返回（function_response）
+ State 变更（state_delta）

### 5.3 Session 管理
#### 创建 Session
```python
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_1",
    session_id="custom-session-id",   # 可选，不传则自动生成 UUID
    state={                            # 可选，初始状态
        "app:model_name": "qwen-max",  # app 级状态（app: 前缀）
        "user:language": "zh-CN",      # user 级状态（user: 前缀）
        "turn_count": 0,               # session 级状态（无前缀）
    },
)
print(f"Session ID: {session.id}")
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `app_name` | str | 是 | 应用名称，对应 OTS 中的 `agent_id` |
| `user_id` | str | 是 | 用户 ID |
| `session_id` | str | 否 | 会话 ID，不传则自动生成 UUID |
| `state` | dict | 否 | 初始状态，会根据 key 前缀自动拆分到三级 State |


> **session_id 的生成策略**：在 Server 场景中，通常由客户端通过 HTTP Header 传入 `session_id`，以便同一用户的多轮对话关联到同一个会话。如果不传，每次请求会创建一个新的独立会话。
>

#### 获取 Session
```python
session = await session_service.get_session(
    app_name="my_app",
    user_id="user_1",
    session_id="your-session-id",
)

if session is None:
    print("会话不存在")
else:
    print(f"事件数: {len(session.events)}")
    print(f"当前状态: {session.state}")
```

返回的 `session` 对象包含：

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `id` | str | 会话 ID |
| `app_name` | str | 应用名称 |
| `user_id` | str | 用户 ID |
| `events` | list[Event] | 完整的 ADK Event 列表（按时间正序） |
| `state` | dict | 合并后的三级状态（详见 5.4 节） |
| `last_update_time` | float | 最后更新时间（Unix 秒级时间戳） |


##### 控制返回的事件数量
当会话事件很多时，可以通过 `GetSessionConfig` 控制只返回最近 N 条事件，避免一次性加载过多数据：

```python
from google.adk.sessions.base_session_service import GetSessionConfig

session = await session_service.get_session(
    app_name="my_app",
    user_id="user_1",
    session_id="your-session-id",
    config=GetSessionConfig(num_recent_events=20),
)
# session.events 只包含最近 20 条事件
```

也可以通过 `after_timestamp` 只返回指定时间之后的事件：

```python
import time

one_hour_ago = time.time() - 3600
session = await session_service.get_session(
    app_name="my_app",
    user_id="user_1",
    session_id="your-session-id",
    config=GetSessionConfig(after_timestamp=one_hour_ago),
)
```

#### 列出 Session
列出指定用户的所有会话，按最后更新时间倒序排列：

```python
response = await session_service.list_sessions(
    app_name="my_app",
    user_id="user_1",
)
for s in response.sessions:
    print(f"Session: {s.id}, 最后更新: {s.last_update_time}")
```

也可以不传 `user_id`，列出该应用下所有用户的会话：

```python
response = await session_service.list_sessions(
    app_name="my_app",
    user_id=None,
)
```

> `list_sessions` 返回的 Session 对象**不包含** events 和 state（出于性能考虑），仅包含元信息。如果需要完整数据，请对感兴趣的 Session 调用 `get_session`。
>

#### 删除 Session
删除会话时会**级联删除**该会话下的所有事件和 Session 级状态：

```python
await session_service.delete_session(
    app_name="my_app",
    user_id="user_1",
    session_id="your-session-id",
)
```

删除顺序为 Event -> State -> Session 元数据。如果中间步骤失败，下次重试可继续清理（幂等安全）。

> 删除 Session 不会影响 App 级和 User 级的状态。
>

#### 同步方法
`OTSSessionService` 的所有方法都提供同步版本，方法名加 `_sync` 后缀：

```python
session = session_service.create_session_sync(
    app_name="my_app", user_id="user_1"
)
session = session_service.get_session_sync(
    app_name="my_app", user_id="user_1", session_id="xxx"
)
response = session_service.list_sessions_sync(
    app_name="my_app", user_id="user_1"
)
session_service.delete_session_sync(
    app_name="my_app", user_id="user_1", session_id="xxx"
)
```

### 5.4 三级 State 机制
Google ADK 定义了三级 State 作用域，Conversation Service 将它们分别持久化到不同的 OTS 表中：

```plain
┌─────────────────────────────────────────────────────────────────────┐
│                        合并后的 session.state                        │
│                                                                     │
│  ┌──────────────────┐                                               │
│  │   App State      │  app_state 表 (agent_id)                      │
│  │   app:model_name │  所有用户、所有会话共享                          │
│  └────────┬─────────┘                                               │
│           │ 覆盖                                                     │
│  ┌────────▼─────────┐                                               │
│  │   User State     │  user_state 表 (agent_id, user_id)            │
│  │   user:language  │  同一用户的所有会话共享                          │
│  └────────┬─────────┘                                               │
│           │ 覆盖                                                     │
│  ┌────────▼─────────┐                                               │
│  │  Session State   │  state 表 (agent_id, user_id, session_id)     │
│  │  turn_count      │  仅当前会话可见                                 │
│  └──────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
```

#### Key 前缀约定
ADK 通过 key 的前缀来区分 State 的作用域：

| 前缀 | 作用域 | 存储位置 | 示例 |
| --- | --- | --- | --- |
| `app:` | App 级 | `app_state` 表 | `app:model_name`、`app:total_queries` |
| `user:` | User 级 | `user_state` 表 | `user:language`、`user:preferences` |
| 无前缀 | Session 级 | `state` 表 | `turn_count`、`last_reply` |
| `temp:` | 临时状态 | 仅内存，不持久化 | `temp:processing` |


#### State 合并规则
当通过 `get_session` 加载会话时，三级 State 会按 **App -> User -> Session** 的顺序浅合并（后者覆盖前者）。返回的 `session.state` 是合并后的完整字典。

例如，如果三级 State 分别为：

```python
# app_state 表
{"model_name": "qwen-max", "version": "1.0"}

# user_state 表
{"language": "zh-CN"}

# state 表 (session)
{"turn_count": 3, "last_reply": "北京今天晴朗"}
```

则 `session.state` 的内容为：

```python
{
    "turn_count": 3,
    "last_reply": "北京今天晴朗",
    "user:language": "zh-CN",
    "app:model_name": "qwen-max",
    "app:version": "1.0",
}
```

#### 通过 state_delta 更新 State
在 ADK 中，Agent 可以通过事件的 `actions.state_delta` 自动更新 State。`OTSSessionService` 会自动将 delta 按前缀拆分并持久化到对应的 State 表：

```python
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

event = Event(
    invocation_id="inv-001",
    author="my_agent",
    content=...,
    actions=EventActions(
        state_delta={
            "turn_count": 1,                # -> state 表
            "app:total_queries": 42,         # -> app_state 表
            "user:last_query_city": "北京",  # -> user_state 表
        },
    ),
)
```

#### output_key 自动写入
ADK Agent 支持 `output_key` 参数，会自动将 Agent 的最终回复写入 `session.state[output_key]`。搭配 `OTSSessionService`，这个值会自动持久化到 OTS 的 Session State 中：

```python
agent = Agent(
    name="assistant",
    model=model,
    instruction="你是一个智能助手。",
    output_key="last_reply",  # Agent 回复自动写入 state["last_reply"]
)
```

后续通过 `get_session` 加载会话时，可以从 `session.state["last_reply"]` 读取上一轮的 Agent 回复。

#### 手动更新 State
除了通过 ADK 的 `state_delta` 自动更新外，也可以直接调用 `SessionStore` 手动更新指定级别的 State。这在 Server 的 `invoke_agent` 回调中常用：

```python
# 更新 Session 级状态
await store.update_session_state_async(
    "my_app", "user_1", "session_id",
    {"turn_count": 5, "last_user_input": "今天天气如何"},
)

# 更新 User 级状态
await store.update_user_state_async(
    "my_app", "user_1",
    {"language": "en-US"},
)

# 更新 App 级状态
await store.update_app_state_async(
    "my_app",
    {"model_name": "qwen-turbo"},
)
```

State 更新采用**浅合并**语义：只覆盖提供的 key，未提供的 key 保持不变。将值设为 `None` 可以删除对应的 key。

### 5.5 结合 AgentRunServer 部署
在生产环境中，通常将 ADK Agent 部署为 HTTP 服务。以下是结合 `AgentRunServer` 的完整示例：

```python
import os
import uuid

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.genai import types

from agentrun import AgentRequest
from agentrun.conversation_service import SessionStore
from agentrun.conversation_service.adapters import OTSSessionService
from agentrun.server import AgentRunServer

APP_NAME = "my_chat_server"

# ── 初始化 ────────────────────────────────────────────────────

store = SessionStore.from_memory_collection(
    os.environ["MEMORY_COLLECTION_NAME"]
)
store.init_tables()

session_service = OTSSessionService(session_store=store)

model = LiteLlm(
    model="openai/qwen3-max",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
agent = Agent(
    name="assistant",
    model=model,
    instruction="你是一个友好的中文智能助手。",
    output_key="last_reply",
)
runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# ── 核心处理函数 ──────────────────────────────────────────────

async def invoke_agent(req: AgentRequest):
    # 从 HTTP Header 获取 session_id 和 user_id
    headers = dict(req.raw_request.headers) if req.raw_request else {}
    session_id = (
        headers.get("x-agentrun-session-id")
        or f"chat_{uuid.uuid4().hex[:8]}"
    )
    user_id = headers.get("x-agentrun-user-id") or "default_user"

    # 获取或创建 Session
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id,
    )
    if session is None:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
        )

    # 提取用户消息
    last_user_text = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            last_user_text = msg.content or ""
            break

    if not last_user_text:
        yield "请输入您的问题。"
        return

    # 调用 ADK Runner 流式输出
    content = types.Content(
        role="user",
        parts=[types.Part(text=last_user_text)],
    )
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if (
            event.is_final_response()
            and event.content
            and event.content.parts
        ):
            for part in event.content.parts:
                if part.text:
                    yield part.text


# ── 启动服务 ──────────────────────────────────────────────────

if __name__ == "__main__":
    server = AgentRunServer(
        invoke_agent=invoke_agent,
        memory_collection_name=os.environ["MEMORY_COLLECTION_NAME"],
    )
    server.start(port=9000)
```

**客户端请求时的 Header 约定：**

| Header | 说明 | 必填 |
| --- | --- | --- |
| `X-AgentRun-Session-ID` | 会话 ID，用于关联多轮对话 | 否（不传则自动生成新会话） |
| `X-AgentRun-User-ID` | 用户 ID | 否（默认 `default_user`） |


---

## 6. LangChain 集成
> Coming Soon — LangChain 适配器 `OTSChatMessageHistory` 正在开发中，将支持与 `RunnableWithMessageHistory` 无缝集成。
>

---

## 7. LangGraph 集成
> Coming Soon — LangGraph 适配器正在规划中。
>

---

## 8. 高级功能
### 8.1 多元索引搜索
Conversation Service 为会话表和状态表创建了多元索引（Search Index），支持不受主键顺序限制的灵活查询。

#### 搜索会话
通过 `SessionStore.search_sessions` 可以按多种条件组合搜索会话：

```python
results, total = store.search_sessions(
    "my_app",
    user_id="user_1",              # 可选，精确匹配用户
    summary_keyword="天气",         # 可选，全文搜索摘要
    labels='["重要"]',             # 可选，精确匹配标签
    framework="adk",               # 可选，精确匹配框架
    updated_after=1700000000000000, # 可选，仅返回此时间后更新的（纳秒时间戳）
    updated_before=None,           # 可选，仅返回此时间前更新的
    is_pinned=True,                # 可选，是否置顶
    limit=20,                      # 每页条数，默认 20
    offset=0,                      # 分页偏移
)

print(f"共 {total} 条结果")
for session in results:
    print(f"  {session.session_id}: {session.summary}")
```

异步版本：

```python
results, total = await store.search_sessions_async("my_app", summary_keyword="天气")
```

#### 按 session_id 独立查询状态
State 表的多元索引（`state_search_index`）支持按 `session_id` 独立精确查询，不需要提供 `agent_id` 和 `user_id` 前缀。这在需要跨用户定位特定会话状态时非常有用。

### 8.2 表名前缀隔离
在多租户场景或需要区分不同环境（开发/测试/生产）时，可以通过 `table_prefix` 参数为所有表名添加前缀：

```python
# 开发环境
dev_store = SessionStore.from_memory_collection(
    "my-collection", table_prefix="dev_"
)
# 表名：dev_conversation, dev_event, dev_state, ...

# 生产环境
prod_store = SessionStore.from_memory_collection(
    "my-collection", table_prefix="prod_"
)
# 表名：prod_conversation, prod_event, prod_state, ...
```

不同前缀的表完全独立，互不影响。

---

## 9. 常见问题
### init_tables() 需要每次启动都调用吗？
可以。`init_tables()` 是**幂等**操作，表或索引已存在时会自动跳过。建议在应用启动时调用，确保所需资源就绪。

### 多元索引创建后多久生效？
多元索引创建后需要数秒到数十秒才能完全生效（取决于数据量）。在索引生效前，`search_sessions` 可能返回不完整的结果。建议首次创建索引后等待几秒再进行搜索操作。

### 为什么 list_sessions 返回的 Session 没有 events 和 state？
这是出于性能考虑。`list_sessions` 用于展示会话列表，只返回元信息（ID、更新时间等）。如果需要某个 Session 的完整事件和状态，请调用 `get_session`。

### Session 删除后 App 级和 User 级状态还在吗？
是的。`delete_session` 只删除 Session 本身及其关联的事件和 Session 级状态。App 级状态和 User 级状态的生命周期独立于单个 Session，不会被级联删除。

### 如何处理并发写入冲突？
State 更新使用**乐观锁**机制（version 字段）。如果两个请求同时更新同一行，后到的请求会因 version 不匹配而失败。在高并发场景下，建议在业务层实现重试逻辑。

### 支持哪些模型？
Conversation Service 不限制模型选择。示例中使用的是通义千问（通过 DashScope API 调用），您可以替换为任何 ADK 支持的模型（如 Gemini、OpenAI 等）。模型选择由 ADK 的 `Agent` 配置决定，与 Conversation Service 无关。

