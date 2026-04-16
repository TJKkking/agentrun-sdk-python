"""Super Agent 控制面辅助函数 / Super Agent Control Plane Helpers

本模块包含:
- 常量: API 版本号 / 协议类型 / 标签 / 资源路径 / RAM 数据域名列表
- URL 工具: ``_add_ram_prefix_to_host`` / ``build_super_agent_endpoint``
- AgentRuntime ↔ SuperAgent 的双向转换:
  ``to_create_input`` / ``to_update_input`` / ``from_agent_runtime``
  / ``is_super_agent`` / ``parse_super_agent_config``
- 为承载 ``externalEndpoint`` 的 Pydantic 与 Dara 层扩展类

不使用模板生成,保持单一来源,避免同步/异步重复维护。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from alibabacloud_agentrun20250910.client import Client as _DaraClient
from alibabacloud_agentrun20250910.models import (
    CreateAgentRuntimeInput as _DaraCreateAgentRuntimeInput,
)
from alibabacloud_agentrun20250910.models import (
    ListAgentRuntimesRequest as _DaraListAgentRuntimesRequest,
)
from alibabacloud_agentrun20250910.models import (
    ProtocolConfiguration as _DaraProtocolConfiguration,
)
from alibabacloud_agentrun20250910.models import (
    UpdateAgentRuntimeInput as _DaraUpdateAgentRuntimeInput,
)
from pydantic import Field

from agentrun.agent_runtime.model import (
    AgentRuntimeArtifact,
    AgentRuntimeContainer,
    AgentRuntimeCreateInput,
    AgentRuntimeProtocolConfig,
    AgentRuntimeUpdateInput,
)
from agentrun.agent_runtime.runtime import AgentRuntime
from agentrun.utils.config import Config
from agentrun.utils.model import NetworkConfig, NetworkMode

# ─── 常量 ─────────────────────────────────────────────
API_VERSION = "2025-09-10"
SUPER_AGENT_PROTOCOL_TYPE = "SUPER_AGENT"
# ``SUPER_AGENT_TAG`` 标识下游 AgentRuntime 是超级 Agent, 用于 list 过滤。
SUPER_AGENT_TAG = "x-agentrun-super-agent"
# ``EXTERNAL_TAG`` 标识下游 AgentRuntime 由外部 (SuperAgent) 托管调用, 不由 AgentRun 直接托管。
EXTERNAL_TAG = "x-agentrun-external"
# 创建下游 AgentRuntime 时固定写入的 tag 列表: ``[EXTERNAL_TAG, SUPER_AGENT_TAG]``。
SUPER_AGENT_CREATE_TAGS = [EXTERNAL_TAG, SUPER_AGENT_TAG]
SUPER_AGENT_RESOURCE_PATH = "__SUPER_AGENT__"
SUPER_AGENT_INVOKE_PATH = "/invoke"
SUPER_AGENT_NAMESPACE = (
    f"{API_VERSION}/super-agents/{SUPER_AGENT_RESOURCE_PATH}"
)

_RAM_DATA_DOMAINS = ("agentrun-data", "funagent-data-pre")


# ─── URL 工具 ──────────────────────────────────────────


def _add_ram_prefix_to_host(base_url: str) -> str:
    """给已知 data host 加 ``-ram`` 前缀.

    仅当 host 第二段命中 :data:`_RAM_DATA_DOMAINS` 时改写为
    ``<host前缀>-ram.<其余>``, 其他情况原样返回。
    与 :meth:`agentrun.utils.data_api.DataAPI._get_ram_data_endpoint` 同源。
    """
    parsed = urlparse(base_url)
    if not parsed.netloc:
        return base_url
    if not any(f".{d}." in parsed.netloc for d in _RAM_DATA_DOMAINS):
        return base_url
    parts = parsed.netloc.split(".", 1)
    if len(parts) != 2:
        return base_url
    ram_netloc = parts[0] + "-ram." + parts[1]
    return urlunparse((
        parsed.scheme,
        ram_netloc,
        parsed.path or "",
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


def build_super_agent_endpoint(cfg: Optional[Config] = None) -> str:
    """构造 ``protocolConfiguration.externalEndpoint`` 的存储值 (不带版本号).

    基于 :meth:`Config.get_data_endpoint` + :func:`_add_ram_prefix_to_host`
    + 追加 ``/super-agents/__SUPER_AGENT__``, 自动适配生产 / 预发 / 自定义网关。
    """
    cfg = Config.with_configs(cfg)
    base = cfg.get_data_endpoint()
    ram_base = _add_ram_prefix_to_host(base)
    return f"{ram_base.rstrip('/')}/super-agents/{SUPER_AGENT_RESOURCE_PATH}"


# ─── Pydantic 扩展类 ────────────────────────────────────
class SuperAgentProtocolConfig(AgentRuntimeProtocolConfig):
    """承载 ``protocol_settings`` + ``external_endpoint`` 的 Pydantic 扩展.

    基类 ``AgentRuntimeProtocolConfig`` 的 ``type`` 字段是 ``HTTP / MCP`` 枚举,
    本子类通过 ``model_construct`` 绕过校验存入字符串 ``"SUPER_AGENT"``。
    """

    protocol_settings: Optional[List[Dict[str, Any]]] = Field(
        alias="protocolSettings", default=None
    )
    external_endpoint: Optional[str] = Field(
        alias="externalEndpoint", default=None
    )


class _SuperAgentCreateInput(AgentRuntimeCreateInput):
    """默认使用 ``serialize_as_any=True`` 的 create input, 保留子类 extras.

    ``external_agent_endpoint_url`` 是基类 ``AgentRuntimeMutableProps`` 没有覆盖
    的顶层字段, 但在 ``x-agentrun-external`` tag 下服务端强制要求填入, 这里显式
    补齐 (alias 由 BaseModel 的 ``to_camel_case`` 生成 → ``externalAgentEndpointUrl``)。
    """

    external_agent_endpoint_url: Optional[str] = None

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(**kwargs)


class _SuperAgentUpdateInput(AgentRuntimeUpdateInput):
    """默认使用 ``serialize_as_any=True`` 的 update input, 保留子类 extras."""

    external_agent_endpoint_url: Optional[str] = None

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(**kwargs)


# ─── Dara 模型猴补丁 ──────────────────────────────────────
# Dara 的 ``ProtocolConfiguration`` 当前版本没有 ``externalEndpoint`` 字段;
# ``AgentRuntimeClient.create_async/update_async`` 内部做
# ``CreateAgentRuntimeInput().from_map(pydantic.model_dump())`` 的 roundtrip,
# 会在 Dara 层丢失此字段。这里做一次加性 patch: 仅追加读写 ``externalEndpoint``,
# 不改变任何现有字段行为, 用模块级哨兵属性保证幂等。

if not getattr(_DaraProtocolConfiguration, "_super_agent_patched", False):
    _orig_to_map = _DaraProtocolConfiguration.to_map
    _orig_from_map = _DaraProtocolConfiguration.from_map

    def _patched_to_map(self: _DaraProtocolConfiguration) -> Dict[str, Any]:
        result = _orig_to_map(self)
        ee = getattr(self, "external_endpoint", None)
        if ee is not None:
            result["externalEndpoint"] = ee
        return result

    def _patched_from_map(
        self: _DaraProtocolConfiguration, m: Optional[Dict[str, Any]] = None
    ) -> _DaraProtocolConfiguration:
        _orig_from_map(self, m)
        if m and m.get("externalEndpoint") is not None:
            self.external_endpoint = m.get("externalEndpoint")
        return self

    _DaraProtocolConfiguration.to_map = _patched_to_map  # type: ignore[assignment]
    _DaraProtocolConfiguration.from_map = _patched_from_map  # type: ignore[assignment]
    _DaraProtocolConfiguration._super_agent_patched = True  # type: ignore[attr-defined]


# Dara 的 ``CreateAgentRuntimeInput`` / ``UpdateAgentRuntimeInput`` 当前版本没有
# ``tags`` 字段, 与 ``ProtocolConfiguration`` 同理会在 Pydantic → Dara 的 roundtrip
# 中被静默丢弃. 这里沿用同款加性 patch, 只补齐 ``tags`` 字段的读写.
def _patch_dara_tags(cls: Any) -> None:
    if getattr(cls, "_super_agent_tags_patched", False):
        return
    _orig_to_map = cls.to_map
    _orig_from_map = cls.from_map

    def _patched_to_map(self: Any) -> Dict[str, Any]:
        result = _orig_to_map(self)
        tags = getattr(self, "tags", None)
        if tags is not None:
            result["tags"] = tags
        return result

    def _patched_from_map(self: Any, m: Optional[Dict[str, Any]] = None) -> Any:
        _orig_from_map(self, m)
        if m and m.get("tags") is not None:
            self.tags = m.get("tags")
        return self

    cls.to_map = _patched_to_map  # type: ignore[assignment]
    cls.from_map = _patched_from_map  # type: ignore[assignment]
    cls._super_agent_tags_patched = True  # type: ignore[attr-defined]


_patch_dara_tags(_DaraCreateAgentRuntimeInput)
_patch_dara_tags(_DaraUpdateAgentRuntimeInput)
# ``ListAgentRuntimesRequest`` 同样没有 ``tags`` 字段: 补上 from_map/to_map 以保留
# 属性; 真正让服务端生效的查询参数注入由下面的 client 级补丁完成。
_patch_dara_tags(_DaraListAgentRuntimesRequest)


# ─── Dara 客户端猴补丁: list 请求 query 注入 tags ───────────────
# 现版 Dara ``Client.list_agent_runtimes_with_options`` 不读 ``request.tags``
# 构造 query, 导致即便 Pydantic 侧把 tags 传下来, 服务端也收不到。这里一次性
# 包裹同步 / 异步两个方法: 若 request 带有 ``tags`` 就在底层 ``call_api`` 调用
# 前把 ``tags`` (列表 → 逗号分隔) 追加到 ``req.query``。
# 每个 API 调用都会 ``_get_client()`` 新建 ``Client`` 实例, 实例属性级别的替换
# 在并发下是安全的。


def _tags_query_value(tags: Any) -> Optional[str]:
    if tags is None:
        return None
    if isinstance(tags, str):
        return tags
    if isinstance(tags, (list, tuple)):
        return ",".join(str(t) for t in tags)
    return str(tags)


def _patch_dara_client_list_tags() -> None:
    if getattr(_DaraClient, "_super_agent_list_tags_patched", False):
        return

    _orig_sync = _DaraClient.list_agent_runtimes_with_options
    _orig_async = _DaraClient.list_agent_runtimes_with_options_async

    def _patched_sync(
        self: Any, request: Any, headers: Any, runtime: Any
    ) -> Any:
        tags_value = _tags_query_value(getattr(request, "tags", None))
        if tags_value is None:
            return _orig_sync(self, request, headers, runtime)
        orig_call_api = self.call_api

        def _injecting(params: Any, req: Any, rt: Any) -> Any:
            if req.query is None:
                req.query = {}
            req.query["tags"] = tags_value
            return orig_call_api(params, req, rt)

        self.call_api = _injecting
        try:
            return _orig_sync(self, request, headers, runtime)
        finally:
            try:
                del self.call_api
            except AttributeError:
                pass

    async def _patched_async(
        self: Any, request: Any, headers: Any, runtime: Any
    ) -> Any:
        tags_value = _tags_query_value(getattr(request, "tags", None))
        if tags_value is None:
            return await _orig_async(self, request, headers, runtime)
        orig_call_api_async = self.call_api_async

        async def _injecting(params: Any, req: Any, rt: Any) -> Any:
            if req.query is None:
                req.query = {}
            req.query["tags"] = tags_value
            return await orig_call_api_async(params, req, rt)

        self.call_api_async = _injecting
        try:
            return await _orig_async(self, request, headers, runtime)
        finally:
            try:
                del self.call_api_async
            except AttributeError:
                pass

    _DaraClient.list_agent_runtimes_with_options = _patched_sync  # type: ignore[assignment]
    _DaraClient.list_agent_runtimes_with_options_async = _patched_async  # type: ignore[assignment]
    _DaraClient._super_agent_list_tags_patched = True  # type: ignore[attr-defined]


_patch_dara_client_list_tags()


# ─── AgentRuntime ↔ SuperAgent 转换 ────────────────────────
def _business_fields_from_args(
    *,
    prompt: Optional[str] = None,
    agents: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    sandboxes: Optional[List[str]] = None,
    workspaces: Optional[List[str]] = None,
    model_service_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """把业务字段 (None 保留为 None) 收拢成 dict, 供 ``protocolSettings.config`` 使用."""
    return {
        "prompt": prompt,
        "agents": agents if agents is not None else [],
        "tools": tools if tools is not None else [],
        "skills": skills if skills is not None else [],
        "sandboxes": sandboxes if sandboxes is not None else [],
        "workspaces": workspaces if workspaces is not None else [],
        "modelServiceName": model_service_name,
        "modelName": model_name,
    }


def _build_protocol_settings_config(
    *, name: str, business: Dict[str, Any]
) -> str:
    """构造 ``protocolSettings[0].config`` 的 JSON 字符串."""
    cfg_dict: Dict[str, Any] = {
        "path": SUPER_AGENT_INVOKE_PATH,
        "prompt": business.get("prompt"),
        "agents": business.get("agents") or [],
        "tools": business.get("tools") or [],
        "skills": business.get("skills") or [],
        "sandboxes": business.get("sandboxes") or [],
        "workspaces": business.get("workspaces") or [],
        "modelServiceName": business.get("modelServiceName"),
        "modelName": business.get("modelName"),
        "metadata": {"agentRuntimeName": name},
    }
    return json.dumps(cfg_dict, ensure_ascii=False)


def _build_protocol_configuration(
    *,
    name: str,
    business: Dict[str, Any],
    cfg: Optional[Config],
) -> SuperAgentProtocolConfig:
    """构造超级 Agent 的 ``protocolConfiguration`` Pydantic 模型."""
    config_json = _build_protocol_settings_config(name=name, business=business)
    settings: List[Dict[str, Any]] = [{
        "type": SUPER_AGENT_PROTOCOL_TYPE,
        "name": name,
        "path": SUPER_AGENT_INVOKE_PATH,
        "config": config_json,
    }]
    pc = SuperAgentProtocolConfig.model_construct(
        type=SUPER_AGENT_PROTOCOL_TYPE,
        protocol_settings=settings,
        external_endpoint=build_super_agent_endpoint(cfg),
    )
    return pc


def to_create_input(
    name: str,
    *,
    description: Optional[str] = None,
    prompt: Optional[str] = None,
    agents: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    sandboxes: Optional[List[str]] = None,
    workspaces: Optional[List[str]] = None,
    model_service_name: Optional[str] = None,
    model_name: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> AgentRuntimeCreateInput:
    """把超级 Agent 业务字段转为 :class:`AgentRuntimeCreateInput`."""
    business = _business_fields_from_args(
        prompt=prompt,
        agents=agents,
        tools=tools,
        skills=skills,
        sandboxes=sandboxes,
        workspaces=workspaces,
        model_service_name=model_service_name,
        model_name=model_name,
    )
    pc = _build_protocol_configuration(name=name, business=business, cfg=cfg)
    # SUPER_AGENT 是平台托管运行时, 不跑用户代码/容器, 但服务端仍要求
    # artifact_type / network_configuration 非空. 这里给占位默认值即可.
    return _SuperAgentCreateInput.model_construct(
        agent_runtime_name=name,
        description=description,
        protocol_configuration=pc,
        tags=list(SUPER_AGENT_CREATE_TAGS),
        # 带 ``x-agentrun-external`` tag 时服务端强制要求 externalAgentEndpointUrl 非空,
        # 对超级 Agent 而言即数据面入口 (与 protocolConfiguration.externalEndpoint 同值)。
        external_agent_endpoint_url=build_super_agent_endpoint(cfg),
        # 占位 artifact: SUPER_AGENT 不跑用户 container/code, 但服务端要求非空。
        artifact_type=AgentRuntimeArtifact.CONTAINER,
        container_configuration=AgentRuntimeContainer(
            image="registry.cn-hangzhou.aliyuncs.com/agentrun/super-agent-placeholder:v1"
        ),
        network_configuration=NetworkConfig(network_mode=NetworkMode.PUBLIC),
    )


def to_update_input(
    name: str,
    merged: Dict[str, Any],
    cfg: Optional[Config] = None,
) -> AgentRuntimeUpdateInput:
    """把合并后的业务字段转为 :class:`AgentRuntimeUpdateInput` (全量替换)."""
    business = _business_fields_from_args(
        prompt=merged.get("prompt"),
        agents=merged.get("agents"),
        tools=merged.get("tools"),
        skills=merged.get("skills"),
        sandboxes=merged.get("sandboxes"),
        workspaces=merged.get("workspaces"),
        model_service_name=merged.get("model_service_name"),
        model_name=merged.get("model_name"),
    )
    pc = _build_protocol_configuration(name=name, business=business, cfg=cfg)
    return _SuperAgentUpdateInput.model_construct(
        agent_runtime_name=name,
        description=merged.get("description"),
        protocol_configuration=pc,
        tags=list(SUPER_AGENT_CREATE_TAGS),
        # 带 ``x-agentrun-external`` tag 时服务端强制要求 externalAgentEndpointUrl 非空。
        external_agent_endpoint_url=build_super_agent_endpoint(cfg),
        # 占位 artifact: SUPER_AGENT 不跑用户 container/code, 但服务端要求非空。
        artifact_type=AgentRuntimeArtifact.CONTAINER,
        container_configuration=AgentRuntimeContainer(
            image="registry.cn-hangzhou.aliyuncs.com/agentrun/super-agent-placeholder:v1"
        ),
        network_configuration=NetworkConfig(network_mode=NetworkMode.PUBLIC),
    )


def _extract_protocol_configuration(rt: AgentRuntime) -> Dict[str, Any]:
    """统一把 rt.protocol_configuration 转为 dict (兼容 dict / pydantic 两种形态)."""
    pc = getattr(rt, "protocol_configuration", None)
    if pc is None:
        return {}
    if isinstance(pc, dict):
        return pc
    # Pydantic 模型: 用 model_dump(serialize_as_any=True) 保留 extras
    try:
        return pc.model_dump(by_alias=True, serialize_as_any=True)
    except TypeError:
        return pc.model_dump(by_alias=True)


def _extract_protocol_settings(pc_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 protocolConfiguration dict 中取出 protocolSettings 列表."""
    for key in ("protocolSettings", "protocol_settings"):
        v = pc_dict.get(key)
        if isinstance(v, list):
            return v
    return []


def is_super_agent(rt: AgentRuntime) -> bool:
    """判断一个 AgentRuntime 是否为超级 Agent."""
    pc_dict = _extract_protocol_configuration(rt)
    if not pc_dict:
        return False
    settings = _extract_protocol_settings(pc_dict)
    if not settings:
        return False
    first = settings[0] if isinstance(settings[0], dict) else {}
    return first.get("type") == SUPER_AGENT_PROTOCOL_TYPE


def parse_super_agent_config(rt: AgentRuntime) -> Dict[str, Any]:
    """反解 ``protocolSettings[0].config`` 为业务字段 dict.

    如果 config 缺失或非法 JSON, 返回空 dict (不抛异常)。
    """
    pc_dict = _extract_protocol_configuration(rt)
    if not pc_dict:
        return {}
    settings = _extract_protocol_settings(pc_dict)
    if not settings:
        return {}
    first = settings[0] if isinstance(settings[0], dict) else {}
    raw_config = first.get("config")
    if not raw_config:
        return {}
    if isinstance(raw_config, dict):
        return raw_config
    if isinstance(raw_config, str):
        try:
            parsed = json.loads(raw_config)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _get_external_endpoint(rt: AgentRuntime) -> str:
    pc_dict = _extract_protocol_configuration(rt)
    return (
        pc_dict.get("externalEndpoint")
        or pc_dict.get("external_endpoint", "")
        or ""
    )


def from_agent_runtime(rt: AgentRuntime) -> "SuperAgent":  # noqa: F821
    """反解 AgentRuntime → SuperAgent 实例 (不注入 ``_client``)."""
    # 延迟导入避免循环
    from agentrun.super_agent.agent import SuperAgent

    business = parse_super_agent_config(rt)
    return SuperAgent(
        name=getattr(rt, "agent_runtime_name", None) or "",
        description=getattr(rt, "description", None),
        prompt=business.get("prompt"),
        agents=list(business.get("agents") or []),
        tools=list(business.get("tools") or []),
        skills=list(business.get("skills") or []),
        sandboxes=list(business.get("sandboxes") or []),
        workspaces=list(business.get("workspaces") or []),
        model_service_name=business.get("modelServiceName"),
        model_name=business.get("modelName"),
        agent_runtime_id=getattr(rt, "agent_runtime_id", None) or "",
        arn=getattr(rt, "agent_runtime_arn", None) or "",
        status=str(getattr(rt, "status", "") or ""),
        created_at=getattr(rt, "created_at", None) or "",
        last_updated_at=getattr(rt, "last_updated_at", None) or "",
        external_endpoint=_get_external_endpoint(rt),
    )


__all__ = [
    "API_VERSION",
    "SUPER_AGENT_PROTOCOL_TYPE",
    "SUPER_AGENT_TAG",
    "EXTERNAL_TAG",
    "SUPER_AGENT_CREATE_TAGS",
    "SUPER_AGENT_RESOURCE_PATH",
    "SUPER_AGENT_INVOKE_PATH",
    "SUPER_AGENT_NAMESPACE",
    "SuperAgentProtocolConfig",
    "_SuperAgentCreateInput",
    "_SuperAgentUpdateInput",
    "build_super_agent_endpoint",
    "_add_ram_prefix_to_host",
    "to_create_input",
    "to_update_input",
    "from_agent_runtime",
    "is_super_agent",
    "parse_super_agent_config",
]
