import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel

if TYPE_CHECKING:
    from questions.base_question import BaseQuestion
    from core.kathara_client import KatharaClient


def _coerce_content_to_text(content) -> str:
    """Normalize an LLM message's ``content`` to a plain string.

    Most providers return a string, but some chat models (e.g. Bedrock
    Converse) return a list of content blocks like
    ``[{"type": "text", "text": "..."}]``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", "") or "")
        return "".join(parts)
    return str(content)


def invoke_structured_with_fallback(structured_llm, messages, output_model, solver=None, fallback_text=""):
    """Invoke a structured-output LLM, recovering the answer from prose on failure.

    ``model.with_structured_output(...).invoke()`` has two failure modes:
      - returns ``None`` (Bedrock Converse drops the structured-output tool call)
      - raises ``OutputParserException`` (Ollama emits malformed JSON)

    In both cases the analyst's preceding prose (``fallback_text``) usually
    already contains the answer as a JSON block, so we parse and validate that
    against ``output_model``. Returns a validated model instance, or ``None`` if
    both the structured call and the fallback fail.

    Imported lazily so this module stays importable without langchain_core.
    """
    from langchain_core.exceptions import OutputParserException
    from pydantic import ValidationError

    try:
        result = structured_llm.invoke(messages)
    except (OutputParserException, ValidationError):
        # OutputParserException: Ollama emits malformed JSON.
        # ValidationError: a Bedrock Converse tool-call returns args that don't
        # match the schema (e.g. omitting an optional-but-required-present key).
        result = None
    if result is not None:
        return result
    # The structured-output tool-call failed (Bedrock Converse routinely drops
    # it). Try recovering the answer from the analyst's prose. Only flag an
    # empty response if that fallback also fails — a successful recovery means
    # the run produced a valid answer and should not be marked empty.
    recovered = BaseSolver.recover_answer_from_text(fallback_text, output_model)
    if recovered is None and solver is not None:
        solver.last_had_empty_response = True
    return recovered


def _iter_json_candidates(text: str):
    """Yield JSON-object substrings found in free text, best candidates first.

    Looks for fenced ```json blocks first (most reliable, since the analyst is
    prompted to emit them), then falls back to brace-balanced scanning so a raw
    object embedded in prose is still recoverable. Candidates are yielded
    last-to-first because the final object in an answer is usually the actual
    answer (earlier ones tend to be intermediate examples).
    """
    if not text:
        return
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    for block in reversed(fenced):
        yield block

    # Brace-balanced scan: collect every top-level {...} region.
    spans = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    spans.append(text[start:i + 1])
    for block in reversed(spans):
        yield block

class BaseSolver(ABC):
    """
    Abstract base class for a system that can solve a given question.

    This abstracts away the difference between a simple LLM call and a
    complex, multi-agent system.
    """

    def __init__(self, lab_path: str):
        """
        Initialize the solver with the lab path.

        Args:
            lab_path (str): Path to the Kathara lab folder
        """
        self.lab_path = lab_path
        self.execution_log: list[dict] = []
        # Per-run diagnostics: populated by subclasses during solve().
        self.last_token_stats: dict = {}
        self.last_had_empty_response: bool = False

    def log_step(self, node: str, step_type: str, **kwargs):
        """
        Record a step in the execution log.

        Args:
            node: Name of the graph node (e.g. "llm_solver", "planner", "tools")
            step_type: Type of step (e.g. "llm_input", "llm_output", "tool_call", "tool_result")
            **kwargs: Additional data to record (content, tool_name, etc.)
        """
        import time
        entry = {
            "timestamp": time.time(),
            "node": node,
            "step_type": step_type,
            **kwargs,
        }
        self.execution_log.append(entry)

    def reset_diagnostics(self) -> None:
        """Reset per-run state. Call at the start of solve()."""
        self.execution_log = []
        self.last_token_stats = {}
        self.last_had_empty_response = False

    @staticmethod
    def recover_answer_from_text(
        text, output_model: "Type[BaseModel]"
    ) -> Optional[BaseModel]:
        """Best-effort extraction of the structured answer from free text.

        Used as a fallback when ``model.with_structured_output(...).invoke()``
        fails to yield a schema-conforming object — either by returning ``None``
        (Bedrock Converse drops the structured-output tool call) or by raising
        ``OutputParserException`` (Ollama emits malformed JSON). In both cases
        the analyst has usually already written the answer as a JSON block in
        its prose response; this parses that block and validates it against the
        question's Pydantic model.

        Returns a validated model instance, or ``None`` if no JSON candidate in
        the text parses and validates. This same function is reused by the
        offline recovery script so recovered runs match the live pipeline.
        """
        plain = _coerce_content_to_text(text).strip()
        if not plain:
            return None
        for candidate in _iter_json_candidates(plain):
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            try:
                return output_model(**data)
            except Exception:
                # Wrong shape (e.g. the model echoed the schema's $defs) — keep
                # trying other candidates rather than accepting an invalid one.
                continue
        return None

    @abstractmethod
    def solve(self, question: "BaseQuestion", model: BaseChatModel) -> BaseModel | None:
        """
        Takes a question and a Kathara client and returns a Pydantic model instance.

        Args:
            question (BaseQuestion): The question to solve.
            model (BaseChatModel): The chat model to use for solving the question.

        Returns:
            BaseModel | None: An instance of the question's output model, or None on failure.
        """
        raise NotImplementedError
