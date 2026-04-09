"""LangGraph BaseCheckpointSaver 适配器。

基于 OTS 的 LangGraph checkpoint 持久化实现，
通过 SessionStore 层访问 checkpoint/checkpoint_writes/checkpoint_blobs 三张表。
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Iterator, Sequence
import json
import logging
import random
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    WRITES_IDX_MAP,
)

from agentrun.conversation_service.session_store import SessionStore

logger = logging.getLogger(__name__)


def _b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64_decode(data: str) -> bytes:
    return base64.b64decode(data)


class OTSCheckpointSaver(BaseCheckpointSaver[str]):
    """基于 OTS 的 LangGraph checkpoint saver。

    将 LangGraph 的 checkpoint 数据持久化到阿里云 TableStore，
    遵循三层架构通过 SessionStore 访问底层存储。

    当指定 ``agent_id`` 时，每次 ``put()`` 会自动在 conversation 表中
    创建/更新会话记录（``session_id = thread_id``，``framework = "langgraph"``），
    使得外部服务可以通过 ``agent_id / user_id`` 查询到 LangGraph 会话列表。

    Args:
        session_store: SessionStore 实例（需已完成 init_checkpoint_tables）。
        agent_id: 智能体 ID。设置后 put 时自动同步 conversation 记录。
        user_id: 默认用户 ID。可通过 ``config["metadata"]["user_id"]``
            在每次调用时覆盖（优先级更高）。

    Example::

        store = await SessionStore.from_memory_collection_async("my-mc")
        await store.init_checkpoint_tables_async()
        checkpointer = OTSCheckpointSaver(
            store, agent_id="my_agent", user_id="default_user"
        )
        # 传入 LangGraph 的 StateGraph.compile(checkpointer=checkpointer)
    """

    store: SessionStore
    agent_id: str
    user_id: str

    def __init__(
        self,
        session_store: SessionStore,
        *,
        agent_id: str = "",
        user_id: str = "",
    ) -> None:
        super().__init__()
        self.store = session_store
        self.agent_id = agent_id
        self.user_id = user_id

    # ------------------------------------------------------------------
    # Version
    # ------------------------------------------------------------------

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dump_typed_b64(self, value: Any) -> tuple[str, str]:
        """序列化值并 base64 编码 bytes 部分。"""
        type_str, data = self.serde.dumps_typed(value)
        return type_str, _b64_encode(data)

    def _load_typed_b64(self, type_str: str, data_b64: str) -> Any:
        """从 base64 编码的字符串反序列化值。"""
        data = _b64_decode(data_b64)
        return self.serde.loads_typed((type_str, data))

    def _load_blobs(
        self,
        blob_map: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        """从 blob 数据重建 channel_values。"""
        channel_values: dict[str, Any] = {}
        for channel, blob_info in blob_map.items():
            blob_type = blob_info.get("blob_type", "")
            blob_data = blob_info.get("blob_data", "")
            if blob_type and blob_type != "empty":
                channel_values[channel] = self._load_typed_b64(
                    blob_type, blob_data
                )
        return channel_values

    def _build_checkpoint_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        row: dict[str, Any],
        blob_map: dict[str, dict[str, str]],
        writes_rows: list[dict[str, Any]],
        config: Optional[RunnableConfig] = None,
    ) -> CheckpointTuple:
        """从存储行数据构建 CheckpointTuple。"""
        checkpoint_id = row["checkpoint_id"]
        checkpoint_type = row.get("checkpoint_type", "")
        checkpoint_data = row.get("checkpoint_data", "")
        metadata_json = row.get("metadata", "{}")
        parent_checkpoint_id = row.get("parent_checkpoint_id", "")

        checkpoint: Checkpoint = self._load_typed_b64(
            checkpoint_type, checkpoint_data
        )
        checkpoint["channel_values"] = self._load_blobs(blob_map)

        metadata: CheckpointMetadata = json.loads(metadata_json)

        pending_writes: list[tuple[str, str, Any]] = []
        for w in writes_rows:
            task_id = w.get("task_id", "")
            channel = w.get("channel", "")
            value_type = w.get("value_type", "")
            value_data = w.get("value_data", "")
            if value_type:
                value = self._load_typed_b64(value_type, value_data)
            else:
                value = None
            pending_writes.append((task_id, channel, value))

        result_config: RunnableConfig = config or {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        parent_config: Optional[RunnableConfig] = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        return CheckpointTuple(
            config=result_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes if pending_writes else None,
        )

    # ------------------------------------------------------------------
    # Session sync helpers
    # ------------------------------------------------------------------

    def _resolve_user_id(self, config: RunnableConfig) -> str:
        """从 config.metadata 或构造器参数中提取 user_id。

        优先级：config["metadata"]["user_id"] > self.user_id > "default"
        """
        md = config.get("metadata") or {}
        if isinstance(md, dict):
            uid = md.get("user_id")
            if uid:
                return str(uid)
        return self.user_id or "default"

    def _sync_session(self, thread_id: str, user_id: str) -> None:
        """同步创建/更新 conversation 表中的会话记录（同步）。"""
        if not self.agent_id:
            return
        try:
            existing = self.store.get_session(self.agent_id, user_id, thread_id)
            if existing is None:
                self.store.create_session(
                    agent_id=self.agent_id,
                    user_id=user_id,
                    session_id=thread_id,
                    framework="langgraph",
                )
            else:
                self.store.update_session(
                    self.agent_id,
                    user_id,
                    thread_id,
                    version=existing.version,
                )
        except Exception:
            logger.warning(
                "Failed to sync conversation record for "
                "agent_id=%s, user_id=%s, thread_id=%s",
                self.agent_id,
                user_id,
                thread_id,
                exc_info=True,
            )

    async def _sync_session_async(self, thread_id: str, user_id: str) -> None:
        """同步创建/更新 conversation 表中的会话记录（异步）。"""
        if not self.agent_id:
            return
        try:
            existing = await self.store.get_session_async(
                self.agent_id, user_id, thread_id
            )
            if existing is None:
                await self.store.create_session_async(
                    agent_id=self.agent_id,
                    user_id=user_id,
                    session_id=thread_id,
                    framework="langgraph",
                )
            else:
                await self.store.update_session_async(
                    self.agent_id,
                    user_id,
                    thread_id,
                    version=existing.version,
                )
        except Exception:
            logger.warning(
                "Failed to sync conversation record for "
                "agent_id=%s, user_id=%s, thread_id=%s",
                self.agent_id,
                user_id,
                thread_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Core: get_tuple
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        row = self.store.get_checkpoint(thread_id, checkpoint_ns, checkpoint_id)
        if row is None:
            return None

        actual_id = row["checkpoint_id"]
        checkpoint_type = row.get("checkpoint_type", "")
        checkpoint_data = row.get("checkpoint_data", "")
        cp: Checkpoint = self._load_typed_b64(checkpoint_type, checkpoint_data)

        blob_map = self.store.get_checkpoint_blobs(
            thread_id, checkpoint_ns, cp.get("channel_versions", {})
        )

        writes_rows = self.store.get_checkpoint_writes(
            thread_id, checkpoint_ns, actual_id
        )

        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": actual_id,
            }
        }

        return self._build_checkpoint_tuple(
            thread_id,
            checkpoint_ns,
            row,
            blob_map,
            writes_rows,
            config=result_config,
        )

    # ------------------------------------------------------------------
    # Core: list
    # ------------------------------------------------------------------

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            return

        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")

        before_id: Optional[str] = None
        if before:
            before_id = get_checkpoint_id(before)

        fetch_limit = limit if limit is not None else 100
        rows = self.store.list_checkpoints(
            thread_id,
            checkpoint_ns,
            limit=fetch_limit,
            before=before_id,
        )

        yielded = 0
        for row in rows:
            if limit is not None and yielded >= limit:
                break

            checkpoint_id = row["checkpoint_id"]
            checkpoint_type = row.get("checkpoint_type", "")
            checkpoint_data = row.get("checkpoint_data", "")
            metadata_json = row.get("metadata", "{}")

            metadata: CheckpointMetadata = json.loads(metadata_json)
            if filter and not all(
                query_value == metadata.get(query_key)
                for query_key, query_value in filter.items()
            ):
                continue

            cp: Checkpoint = self._load_typed_b64(
                checkpoint_type, checkpoint_data
            )

            blob_map = self.store.get_checkpoint_blobs(
                thread_id, checkpoint_ns, cp.get("channel_versions", {})
            )

            writes_rows = self.store.get_checkpoint_writes(
                thread_id, checkpoint_ns, checkpoint_id
            )

            yield self._build_checkpoint_tuple(
                thread_id, checkpoint_ns, row, blob_map, writes_rows
            )
            yielded += 1

    # ------------------------------------------------------------------
    # Core: put
    # ------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id", "")

        c = checkpoint.copy()
        channel_values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]

        for channel, version in new_versions.items():
            if channel in channel_values:
                blob_type, blob_data = self._dump_typed_b64(
                    channel_values[channel]
                )
            else:
                blob_type, blob_data = "empty", ""

            self.store.put_checkpoint_blob(
                thread_id,
                checkpoint_ns,
                channel,
                str(version),
                blob_type=blob_type,
                blob_data=blob_data,
            )

        cp_type, cp_data = self._dump_typed_b64(c)
        final_metadata = get_checkpoint_metadata(config, metadata)
        metadata_json = json.dumps(final_metadata, ensure_ascii=False)

        self.store.put_checkpoint(
            thread_id,
            checkpoint_ns,
            checkpoint["id"],
            checkpoint_type=cp_type,
            checkpoint_data=cp_data,
            metadata_json=metadata_json,
            parent_checkpoint_id=parent_checkpoint_id or "",
        )

        self._sync_session(thread_id, self._resolve_user_id(config))

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    # ------------------------------------------------------------------
    # Core: put_writes
    # ------------------------------------------------------------------

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        existing_writes = self.store.get_checkpoint_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        existing_keys: set[str] = set()
        for w in existing_writes:
            existing_keys.add(w.get("task_idx", ""))

        write_rows: list[dict[str, Any]] = []
        for idx, (channel, value) in enumerate(writes):
            mapped_idx = WRITES_IDX_MAP.get(channel, idx)
            task_idx = f"{task_id}:{mapped_idx}"

            if mapped_idx >= 0 and task_idx in existing_keys:
                continue

            value_type, value_data = self._dump_typed_b64(value)

            write_rows.append({
                "task_idx": task_idx,
                "task_id": task_id,
                "task_path": task_path,
                "channel": channel,
                "value_type": value_type,
                "value_data": value_data,
            })

        if write_rows:
            self.store.put_checkpoint_writes(
                thread_id, checkpoint_ns, checkpoint_id, write_rows
            )

    # ------------------------------------------------------------------
    # Core: delete_thread
    # ------------------------------------------------------------------

    def delete_thread(self, thread_id: str) -> None:
        self.store.delete_thread_checkpoints(thread_id)
        if self.agent_id:
            user_id = self.user_id or "default"
            try:
                self.store.delete_session(self.agent_id, user_id, thread_id)
            except Exception:
                logger.warning(
                    "Failed to delete conversation record for "
                    "agent_id=%s, user_id=%s, thread_id=%s",
                    self.agent_id,
                    user_id,
                    thread_id,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Async versions
    # ------------------------------------------------------------------

    async def aget_tuple(
        self, config: RunnableConfig
    ) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        row = await self.store.get_checkpoint_async(
            thread_id, checkpoint_ns, checkpoint_id
        )
        if row is None:
            return None

        actual_id = row["checkpoint_id"]
        checkpoint_type = row.get("checkpoint_type", "")
        checkpoint_data = row.get("checkpoint_data", "")
        cp: Checkpoint = self._load_typed_b64(checkpoint_type, checkpoint_data)

        blob_map = await self.store.get_checkpoint_blobs_async(
            thread_id, checkpoint_ns, cp.get("channel_versions", {})
        )

        writes_rows = await self.store.get_checkpoint_writes_async(
            thread_id, checkpoint_ns, actual_id
        )

        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": actual_id,
            }
        }

        return self._build_checkpoint_tuple(
            thread_id,
            checkpoint_ns,
            row,
            blob_map,
            writes_rows,
            config=result_config,
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is None:
            return

        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")

        before_id: Optional[str] = None
        if before:
            before_id = get_checkpoint_id(before)

        fetch_limit = limit if limit is not None else 100
        rows = await self.store.list_checkpoints_async(
            thread_id,
            checkpoint_ns,
            limit=fetch_limit,
            before=before_id,
        )

        yielded = 0
        for row in rows:
            if limit is not None and yielded >= limit:
                break

            checkpoint_id = row["checkpoint_id"]
            checkpoint_type = row.get("checkpoint_type", "")
            checkpoint_data = row.get("checkpoint_data", "")
            metadata_json = row.get("metadata", "{}")

            metadata: CheckpointMetadata = json.loads(metadata_json)
            if filter and not all(
                query_value == metadata.get(query_key)
                for query_key, query_value in filter.items()
            ):
                continue

            cp: Checkpoint = self._load_typed_b64(
                checkpoint_type, checkpoint_data
            )

            blob_map = await self.store.get_checkpoint_blobs_async(
                thread_id, checkpoint_ns, cp.get("channel_versions", {})
            )

            writes_rows = await self.store.get_checkpoint_writes_async(
                thread_id, checkpoint_ns, checkpoint_id
            )

            yield self._build_checkpoint_tuple(
                thread_id, checkpoint_ns, row, blob_map, writes_rows
            )
            yielded += 1

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id", "")

        c = checkpoint.copy()
        channel_values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]

        for channel, version in new_versions.items():
            if channel in channel_values:
                blob_type, blob_data = self._dump_typed_b64(
                    channel_values[channel]
                )
            else:
                blob_type, blob_data = "empty", ""

            await self.store.put_checkpoint_blob_async(
                thread_id,
                checkpoint_ns,
                channel,
                str(version),
                blob_type=blob_type,
                blob_data=blob_data,
            )

        cp_type, cp_data = self._dump_typed_b64(c)
        final_metadata = get_checkpoint_metadata(config, metadata)
        metadata_json = json.dumps(final_metadata, ensure_ascii=False)

        await self.store.put_checkpoint_async(
            thread_id,
            checkpoint_ns,
            checkpoint["id"],
            checkpoint_type=cp_type,
            checkpoint_data=cp_data,
            metadata_json=metadata_json,
            parent_checkpoint_id=parent_checkpoint_id or "",
        )

        await self._sync_session_async(thread_id, self._resolve_user_id(config))

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        existing_writes = await self.store.get_checkpoint_writes_async(
            thread_id, checkpoint_ns, checkpoint_id
        )
        existing_keys: set[str] = set()
        for w in existing_writes:
            existing_keys.add(w.get("task_idx", ""))

        write_rows: list[dict[str, Any]] = []
        for idx, (channel, value) in enumerate(writes):
            mapped_idx = WRITES_IDX_MAP.get(channel, idx)
            task_idx = f"{task_id}:{mapped_idx}"

            if mapped_idx >= 0 and task_idx in existing_keys:
                continue

            value_type, value_data = self._dump_typed_b64(value)

            write_rows.append({
                "task_idx": task_idx,
                "task_id": task_id,
                "task_path": task_path,
                "channel": channel,
                "value_type": value_type,
                "value_data": value_data,
            })

        if write_rows:
            await self.store.put_checkpoint_writes_async(
                thread_id, checkpoint_ns, checkpoint_id, write_rows
            )

    async def adelete_thread(self, thread_id: str) -> None:
        await self.store.delete_thread_checkpoints_async(thread_id)
        if self.agent_id:
            user_id = self.user_id or "default"
            try:
                await self.store.delete_session_async(
                    self.agent_id, user_id, thread_id
                )
            except Exception:
                logger.warning(
                    "Failed to delete conversation record for "
                    "agent_id=%s, user_id=%s, thread_id=%s",
                    self.agent_id,
                    user_id,
                    thread_id,
                    exc_info=True,
                )
