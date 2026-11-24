from __future__ import annotations

from ..config import LLMConfig
from ..workspace import Workspace
from .base import AgentResult, AgentTool, LLMClient, LLMError, LLMUnavailable, NullLLM
from .replay import ReplayLLM, ReplayMissing

__all__ = [
    "AgentResult",
    "AgentTool",
    "LLMClient",
    "LLMError",
    "LLMUnavailable",
    "NullLLM",
    "ReplayLLM",
    "ReplayMissing",
    "make_llm",
]


def make_llm(cfg: LLMConfig, workspace: Workspace | None = None) -> LLMClient:
    if cfg.provider == "none":
        return NullLLM(workspace)
    if cfg.provider == "replay":
        if cfg.replay_dir is None:
            raise ValueError("llm.replay_dir is required when llm.provider is 'replay'")
        return ReplayLLM(cfg.replay_dir, workspace)
    if cfg.provider == "openai_compatible":
        from .openai_compatible import OpenAICompatibleLLM

        return OpenAICompatibleLLM(cfg, workspace)
    from .anthropic_client import AnthropicLLM

    return AnthropicLLM(cfg, workspace)
