"""
Token usage callback for LangChain LLM calls.

Aggregates input/output/total tokens across every LLM invocation that runs
within a solver's workflow. Use by attaching to `config={'callbacks': [cb]}`
when invoking a compiled LangGraph app.

Most LangChain chat models populate `response.usage_metadata` with
`{input_tokens, output_tokens, total_tokens}`. We read it from
`on_llm_end` / `on_chat_model_end` events.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class TokenUsageCallback(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls during a solver run."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.n_calls = 0
        # When usage_metadata is absent we still count the call.
        self.n_calls_without_usage = 0

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.n_calls = 0
        self.n_calls_without_usage = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "n_llm_calls": self.n_calls,
            "n_calls_without_usage": self.n_calls_without_usage,
        }

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401
        self._consume(response)

    def on_chat_model_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401
        self._consume(response)

    def _consume(self, response: Any) -> None:
        self.n_calls += 1
        try:
            generations = getattr(response, "generations", None) or []
            found_usage = False
            for gen_group in generations:
                for gen in gen_group:
                    msg = getattr(gen, "message", None)
                    usage = getattr(msg, "usage_metadata", None) if msg else None
                    if not usage:
                        continue
                    self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                    self.output_tokens += int(usage.get("output_tokens", 0) or 0)
                    self.total_tokens += int(usage.get("total_tokens", 0) or 0)
                    found_usage = True
            if not found_usage:
                # Fall back to llm_output (older providers stash totals there).
                llm_output = getattr(response, "llm_output", None) or {}
                token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
                if token_usage:
                    self.input_tokens += int(token_usage.get("prompt_tokens", 0) or 0)
                    self.output_tokens += int(token_usage.get("completion_tokens", 0) or 0)
                    self.total_tokens += int(token_usage.get("total_tokens", 0) or 0)
                    found_usage = True
            if not found_usage:
                self.n_calls_without_usage += 1
        except Exception as e:
            logger.debug(f"TokenUsageCallback failed to read usage: {e}")
            self.n_calls_without_usage += 1
