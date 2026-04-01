"""Tests for agentrun.sandbox.custom_sandbox module."""

from agentrun.sandbox.custom_sandbox import CustomSandbox
from agentrun.sandbox.model import TemplateType


class TestCustomSandbox:

    def test_template_type(self):
        assert (
            CustomSandbox.__private_attributes__["_template_type"].default
            == TemplateType.CUSTOM
        )

    def test_get_base_url(self):
        from agentrun.utils.config import Config

        cfg = Config(
            data_endpoint=(
                "https://account123.agentrun-data.cn-hangzhou.aliyuncs.com"
            )
        )
        sb = CustomSandbox.model_construct(sandbox_id="sb-1")
        result = sb.get_base_url(config=cfg)
        assert (
            result
            == "https://account123.agentrun-data.cn-hangzhou.aliyuncs.com/sandboxes"
        )
        assert "-ram" not in result

    def test_get_base_url_with_config(self):
        from agentrun.utils.config import Config

        cfg = Config(data_endpoint="https://custom.com")
        sb = CustomSandbox.model_construct(sandbox_id="sb-1")
        result = sb.get_base_url(config=cfg)
        assert result == "https://custom.com/sandboxes"
