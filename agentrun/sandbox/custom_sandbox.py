from typing import Optional

from agentrun.sandbox.model import TemplateType
from agentrun.utils.config import Config

from .sandbox import Sandbox


class CustomSandbox(Sandbox):
    """Custom Sandbox"""

    _template_type = TemplateType.CUSTOM

    def get_base_url(self, config: Optional[Config] = None):
        """Get the base URL for the custom sandbox template.

        Returns the non-RAM data endpoint so that the URL can be used
        directly without requiring RAM signature headers.
        返回非 RAM 的数据端点，以便用户可以直接使用该 URL 而无需附带 RAM 签名头。
        """
        cfg = Config.with_configs(config)
        base = cfg.get_data_endpoint()
        return "/".join(part.strip("/") for part in [base, "sandboxes"] if part)
