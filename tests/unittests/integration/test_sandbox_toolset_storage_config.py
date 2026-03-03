"""SandboxToolSet 存储挂载配置透传单元测试

Tests that oss_mount_config, nas_config, and polar_fs_config are correctly
stored and passed through to Sandbox.create() in the ToolSet hierarchy.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from agentrun.integration.builtin.sandbox import (
    BrowserToolSet,
    CodeInterpreterToolSet,
    sandbox_toolset,
    SandboxToolSet,
)
from agentrun.sandbox.model import (
    NASConfig,
    NASMountConfig,
    OSSMountConfig,
    OSSMountPoint,
    PolarFsConfig,
    PolarFsMountConfig,
    TemplateType,
)


@pytest.fixture
def oss_config():
    return OSSMountConfig(
        mount_points=[
            OSSMountPoint(
                bucket_name="test-bucket",
                bucket_path="/data",
                endpoint="oss-cn-hangzhou.aliyuncs.com",
                mount_dir="/mnt/oss",
                read_only=False,
            )
        ]
    )


@pytest.fixture
def nas_config():
    return NASConfig(
        group_id=1000,
        user_id=1000,
        mount_points=[
            NASMountConfig(
                enable_tls=True,
                mount_dir="/mnt/nas",
                server_addr="file-system-id.cn-hangzhou.nas.aliyuncs.com",
            )
        ],
    )


@pytest.fixture
def polar_fs_config():
    return PolarFsConfig(
        group_id=1000,
        user_id=1000,
        mount_points=[
            PolarFsMountConfig(
                instance_id="polar-instance-001",
                mount_dir="/mnt/polar",
                remote_dir="/shared",
            )
        ],
    )


class TestSandboxToolSetStorageConfig:
    """Test that SandboxToolSet base class stores storage configs correctly."""

    def test_stores_all_storage_configs(
        self, oss_config, nas_config, polar_fs_config
    ):
        with patch.object(SandboxToolSet, "__init__", lambda self, **kw: None):
            ts = SandboxToolSet.__new__(SandboxToolSet)

        SandboxToolSet.__init__(
            ts,
            template_name="test-tpl",
            template_type=TemplateType.CODE_INTERPRETER,
            sandbox_idle_timeout_seconds=600,
            config=None,
            oss_mount_config=oss_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )

        assert ts.oss_mount_config is oss_config
        assert ts.nas_config is nas_config
        assert ts.polar_fs_config is polar_fs_config

    def test_defaults_to_none(self):
        with patch.object(SandboxToolSet, "__init__", lambda self, **kw: None):
            ts = SandboxToolSet.__new__(SandboxToolSet)

        SandboxToolSet.__init__(
            ts,
            template_name="test-tpl",
            template_type=TemplateType.CODE_INTERPRETER,
            sandbox_idle_timeout_seconds=600,
            config=None,
        )

        assert ts.oss_mount_config is None
        assert ts.nas_config is None
        assert ts.polar_fs_config is None


class TestEnsureSandboxPassthrough:
    """Test that _ensure_sandbox passes storage configs to Sandbox.create()."""

    @patch("agentrun.integration.builtin.sandbox.Sandbox")
    def test_passes_all_storage_configs(
        self, mock_sandbox_cls, oss_config, nas_config, polar_fs_config
    ):
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "sb-123"
        mock_sandbox_cls.create.return_value = mock_sb

        ts = CodeInterpreterToolSet(
            template_name="test-tpl",
            config=None,
            sandbox_idle_timeout_seconds=600,
            oss_mount_config=oss_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )

        ts._ensure_sandbox()

        mock_sandbox_cls.create.assert_called_once_with(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-tpl",
            sandbox_idle_timeout_seconds=600,
            oss_mount_config=oss_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
            config=None,
        )

    @patch("agentrun.integration.builtin.sandbox.Sandbox")
    def test_passes_none_when_not_provided(self, mock_sandbox_cls):
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "sb-456"
        mock_sandbox_cls.create.return_value = mock_sb

        ts = CodeInterpreterToolSet(
            template_name="test-tpl",
            config=None,
            sandbox_idle_timeout_seconds=600,
        )

        ts._ensure_sandbox()

        mock_sandbox_cls.create.assert_called_once_with(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-tpl",
            sandbox_idle_timeout_seconds=600,
            oss_mount_config=None,
            nas_config=None,
            polar_fs_config=None,
            config=None,
        )


class TestCodeInterpreterToolSetStorageConfig:
    """Test CodeInterpreterToolSet correctly passes storage configs to base."""

    def test_passes_storage_configs_to_base(
        self, oss_config, nas_config, polar_fs_config
    ):
        ts = CodeInterpreterToolSet(
            template_name="ci-tpl",
            config=None,
            sandbox_idle_timeout_seconds=300,
            oss_mount_config=oss_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )

        assert ts.oss_mount_config is oss_config
        assert ts.nas_config is nas_config
        assert ts.polar_fs_config is polar_fs_config
        assert ts.template_type == TemplateType.CODE_INTERPRETER

    def test_backward_compatible_without_storage_configs(self):
        ts = CodeInterpreterToolSet(
            template_name="ci-tpl",
            config=None,
            sandbox_idle_timeout_seconds=300,
        )

        assert ts.oss_mount_config is None
        assert ts.nas_config is None
        assert ts.polar_fs_config is None


class TestBrowserToolSetStorageConfig:
    """Test BrowserToolSet correctly passes storage configs to base."""

    def test_passes_storage_configs_to_base(
        self, oss_config, nas_config, polar_fs_config
    ):
        ts = BrowserToolSet(
            template_name="br-tpl",
            config=None,
            sandbox_idle_timeout_seconds=300,
            oss_mount_config=oss_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )

        assert ts.oss_mount_config is oss_config
        assert ts.nas_config is nas_config
        assert ts.polar_fs_config is polar_fs_config
        assert ts.template_type == TemplateType.BROWSER

    def test_backward_compatible_without_storage_configs(self):
        ts = BrowserToolSet(
            template_name="br-tpl",
            config=None,
            sandbox_idle_timeout_seconds=300,
        )

        assert ts.oss_mount_config is None
        assert ts.nas_config is None
        assert ts.polar_fs_config is None


class TestSandboxToolsetFactory:
    """Test sandbox_toolset factory function passes storage configs."""

    def test_code_interpreter_with_storage_configs(
        self, oss_config, nas_config, polar_fs_config
    ):
        ts = sandbox_toolset(
            "ci-tpl",
            template_type=TemplateType.CODE_INTERPRETER,
            sandbox_idle_timeout_seconds=600,
            oss_mount_config=oss_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )

        assert isinstance(ts, CodeInterpreterToolSet)
        assert ts.oss_mount_config is oss_config
        assert ts.nas_config is nas_config
        assert ts.polar_fs_config is polar_fs_config

    def test_browser_with_storage_configs(
        self, oss_config, nas_config, polar_fs_config
    ):
        ts = sandbox_toolset(
            "br-tpl",
            template_type=TemplateType.BROWSER,
            sandbox_idle_timeout_seconds=600,
            oss_mount_config=oss_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )

        assert isinstance(ts, BrowserToolSet)
        assert ts.oss_mount_config is oss_config
        assert ts.nas_config is nas_config
        assert ts.polar_fs_config is polar_fs_config

    def test_factory_backward_compatible(self):
        ts = sandbox_toolset(
            "ci-tpl",
            template_type=TemplateType.CODE_INTERPRETER,
        )

        assert isinstance(ts, CodeInterpreterToolSet)
        assert ts.oss_mount_config is None
        assert ts.nas_config is None
        assert ts.polar_fs_config is None
