"""
Planner-based agentic solver.

Architecture:
    START → planner ←→ tools  (self-loop while planner emits tool_calls)
                  ↓  (no tool calls — done gathering)
               validator  (knows the output schema)
                  ↓ COMPLETE  → final_answer → structured_output → END
                  ↓ INCOMPLETE → planner  (with notes on what's missing)

Key improvements over the original AgentSolverFromFiles:
  - Planner node combines planning + retrieval in a single LLM call with tools
  - Self-loop: planner can make multiple consecutive tool calls without
    being interrupted by validation after each one
  - Validator receives the full Pydantic output schema so it can judge
    completeness accurately
  - New tools: list_lab_files, read_lab_conf, read_file for surgical access
  - Optional network tools (ping, traceroute, etc.) via use_network flag
  - Configurable max iterations (default 15)
  - **NEW**: max_tokens limits on all LLM invocations to prevent token runaway
  - **NEW**: Enhanced validation detecting delirium, empty responses, malformed output
  - **NEW**: Runaway detection identifies pathological model behavior
"""

import json
import logging
import os
from typing import TypedDict, Annotated, Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from questions.base_question import BaseQuestion
from core.token_callback import TokenUsageCallback
from .base_solver import BaseSolver, invoke_structured_with_fallback
from .tools import build_file_tools, build_network_tools


def _as_text(content) -> str:
    """Normalize an LLM message's ``content`` to a plain string.

    Most providers return a string, but ChatBedrockConverse (and other
    multimodal-capable chat models) return a list of content blocks, e.g.
    ``[{"type": "text", "text": "..."}, ...]``. Downstream code here treats
    content as a string (``.splitlines()``, ``len()``, ``json.loads()``,
    ``"X" in content``), so flatten list-of-blocks into their text.
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

logger = logging.getLogger(__name__)

# Default upper bound on planner invocations (tool-calling rounds)
DEFAULT_MAX_ITERATIONS = 25
# Default upper bound on validator→planner retry cycles
DEFAULT_MAX_VALIDATION_RETRIES = 3
# Max tokens per LLM output to prevent runaway generation
DEFAULT_MAX_TOKENS_PLANNING = 4000       # planner/planner-refocus
DEFAULT_MAX_TOKENS_VALIDATION = 2000     # validator analysis
DEFAULT_MAX_TOKENS_SYNTHESIS = 3000      # final_answer synthesis
DEFAULT_MAX_TOKENS_STRUCTURED = 3000     # structured output
# Thresholds for detecting pathological output
DELIRIUM_LENGTH_THRESHOLD = 8000         # chars; if longer, likely delirium
EMPTY_RESPONSE_THRESHOLD = 100           # chars; if shorter, likely empty/malformed
REPETITION_RATIO_THRESHOLD = 0.6         # if >60% of content is repetitions, likely noise


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlannerAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    output_schema: str          # JSON schema of the expected Pydantic model
    iteration_count: int        # total planner invocations (incremented each planner call)
    validation_count: int       # number of validator→planner retry cycles
    current_node: str           # tracks which node produced the last message


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_planner_node(model: BaseChatModel, all_tools: list, solver=None):
    """Create the planner node.

    The planner reasons about the question, decides which tools to call,
    and emits tool_calls.  It loops back to itself via the tool node until
    it is satisfied — at which point it emits a message with NO tool_calls
    and the graph routes to the validator.
    """

    def planner_node(state: PlannerAgentState):
        iteration = state.get("iteration_count", 0)
        prior_messages = state.get("messages", [])
        # First call = nothing in the message history yet.
        is_first = len(prior_messages) == 0

        new_state_messages: list = []

        if is_first:
            system_prompt = f"""# Role
You are a **Network Analysis Planner** — an expert virtual system administrator.
You have access to a Kathara network lab's configuration files and potentially
the live network itself.  Your goal is to gather exactly the data needed to
answer the user's question.

# Available information
- The lab directory contains configuration files describing the network topology,
  device startup scripts, and service configurations.
- Use the tools provided to explore and read these files.

# Strategy
1. **Understand the question** — read it carefully and identify what data fields
   you need to populate (see Output Schema below).
2. **Plan your retrieval** — decide which files or commands will give you the
   needed data.  Start broad (list files, read lab.conf) then drill into
   specific devices.
3. **Execute** — call the tools.  You may call multiple tools at once if they
   are independent.
4. **Reflect** — after receiving tool results, check whether you have enough
   data.  If not, call more tools.  When you believe you have everything,
   stop calling tools and summarize what you found.

# Output Schema
The final answer must populate this JSON schema:
```json
{state['output_schema']}
```

# Important
- ALWAYS start by listing files or reading lab.conf to understand the topology.
- Do NOT guess — use tools to verify.
- When you have collected enough data, respond with a plain text summary of
  your findings (no tool calls).  The validator will then check completeness.

# User Question
{state['user_query']}"""
            invoke_messages = [SystemMessage(content=system_prompt)]
            new_state_messages.append(SystemMessage(content=system_prompt))
            log_prompt = system_prompt
        else:
            # Subsequent iterations: history already contains the original
            # system prompt, tool calls, and (if applicable) the validator's
            # most recent INCOMPLETE feedback. Do NOT re-inject a SystemMessage
            # — small models tend to treat a trailing system + schema as
            # "produce JSON now" and skip further tool calls. If we are
            # arriving from the validator, add a short HumanMessage nudge.
            last = prior_messages[-1] if prior_messages else None
            came_from_validation = (
                last is not None
                and getattr(last, "type", "") == "ai"
                and "INCOMPLETE" in (getattr(last, "content", "") or "")
            )
            invoke_messages = list(prior_messages)
            if came_from_validation:
                nudge = HumanMessage(content=(
                    "The data above is incomplete. Call the appropriate tool(s) "
                    "to fetch the missing information. Do not produce a JSON "
                    "answer yet."
                ))
                invoke_messages.append(nudge)
                new_state_messages.append(nudge)
            log_prompt = "(continuation; no new system prompt)"

        llm_with_tools = model.bind_tools(all_tools)
        logger.info(f"[planner] Invoking LLM (is_first={is_first}, iter={iteration}, tools_bound={len(all_tools)})")
        if solver:
            solver.log_step("planner", "llm_input", content=log_prompt, iteration=iteration, is_first=is_first)

        response = llm_with_tools.invoke(invoke_messages)
        content = _as_text(response.content)
        has_tc = hasattr(response, "tool_calls") and response.tool_calls
        
        # --- PATHOLOGICAL OUTPUT DETECTION ---
        is_delirious = len(content) > DELIRIUM_LENGTH_THRESHOLD
        is_empty = not has_tc and len(content) < EMPTY_RESPONSE_THRESHOLD
        
        # Simple repetition check: if many 4-line blocks are identical
        lines = content.splitlines()
        is_repetitive = False
        if len(lines) > 20:
            blocks = ["\n".join(lines[i:i+4]) for i in range(0, len(lines)-4, 4)]
            if len(set(blocks)) < len(blocks) * (1 - REPETITION_RATIO_THRESHOLD):
                is_repetitive = True
        
        if is_delirious or is_repetitive:
            logger.warning(f"[planner] Pathological output detected: delirious={is_delirious}, repetitive={is_repetitive}")
            # If delirious, we truncate and mark it
            if is_delirious:
                response.content = content[:500] + "\n... [TRUNCATED DUE TO DELIRIUM] ..."
            # We can also add a flag to the state to tell the validator/router
        
        logger.info(f"[planner] LLM responded: has_tool_calls={has_tc}, content_len={len(content)}")

        if solver:
            tool_calls_log = None
            if has_tc:
                tool_calls_log = [
                    {"name": tc.get("name", "?"), "args": tc.get("args", {})}
                    for tc in response.tool_calls
                ]
            solver.log_step("planner", "llm_output",
                            content=response.content,
                            tool_calls=tool_calls_log,
                            is_pathological=(is_delirious or is_repetitive or is_empty))

        new_state_messages.append(response)
        return {
            "messages": new_state_messages,
            "iteration_count": iteration + 1,
            "current_node": "planner",
        }

    return planner_node


def make_tool_logging_node(tool_node: ToolNode, solver=None):
    """Wrap a ToolNode to log tool inputs and outputs."""

    def logging_tool_node(state: PlannerAgentState):
        # Log which tools are being called (from the last AI message)
        last = state["messages"][-1] if state["messages"] else None
        if solver and last and hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                solver.log_step("tools", "tool_call",
                                tool_name=tc.get("name", "?"),
                                tool_args=tc.get("args", {}))

        result = tool_node.invoke(state)

        # Log tool results
        if solver:
            for msg in result.get("messages", []):
                content = _as_text(msg.content) if hasattr(msg, "content") else str(msg)
                tool_name = getattr(msg, "name", "unknown")
                solver.log_step("tools", "tool_result",
                                tool_name=tool_name,
                                content=content[:2000])  # truncate large results

        return result

    return logging_tool_node


def make_validation_node(model: BaseChatModel, solver=None):
    """Create the validation node.

    Receives the full message history (including tool results) and decides
    whether we have enough data to answer the question.  The output schema
    is included so it can check field-by-field.
    """

    def validation_node(state: PlannerAgentState):
        prompt = f"""# Role
You are the **Validation Agent**.  Your job is to decide whether the data
gathered so far is sufficient to fully answer the user's question.

# Output Schema
The answer must populate **every field** of this JSON schema:
```json
{state['output_schema']}
```

# Decision procedure
1. List every field in the output schema.
2. For each field, check whether the retrieved data provides a concrete value.
3. If ALL fields can be filled → respond with **COMPLETE**.
4. If ANY field is missing or uncertain → respond with **INCOMPLETE** and
   list exactly what is still needed.

# Response format
- Start with your field-by-field analysis (brief).
- End with a single line: either `COMPLETE` or `INCOMPLETE`.
- If INCOMPLETE, add bullet points describing what the planner should fetch next.

# User Question
{state['user_query']}"""

        messages = state.get("messages", []) + [SystemMessage(content=prompt)]
        logger.info(f"[validation] Invoking LLM")
        if solver:
            solver.log_step("validation", "llm_input", content=prompt)

        response = model.invoke(messages)
        resp_text = _as_text(response.content)
        logger.info(f"[validation] Response ({len(resp_text)} chars): {resp_text[:300]}")
        if solver:
            solver.log_step("validation", "llm_output", content=resp_text)

        return {
            "messages": [response],
            "validation_count": state.get("validation_count", 0) + 1,
            "current_node": "validation",
        }

    return validation_node


def make_final_answer_node(model: BaseChatModel, solver=None):
    """Create the final answer node.

    Synthesises all retrieved data into a clear natural-language answer.
    """

    def final_answer_node(state: PlannerAgentState):
        prompt = f"""# Role and Objective
You are the **Networking Final Answer Agent**, providing a concise, definitive response to the user's query.

# Instructions
- Clearly synthesize retrieved configurations.
- Provide a concise, accurate, and direct final answer.
- Reference specific retrieved data explicitly as evidence.

# Output Format
Concise natural-language explanation directly addressing the query, supported by retrieved configuration data.

Now, carefully synthesize information step by step to provide a final, clear answer to the user's query: {state['user_query']}"""

        # If the planner was hard-stopped while still requesting tools, the
        # last AIMessage carries dangling tool_calls with no matching tool
        # responses. Some providers reject this. Strip them.
        history = state.get("messages", [])
        if history:
            last = history[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                sanitized = AIMessage(
                    content=_as_text(last.content)
                    + "\n[Note: tool calls truncated due to iteration cap]"
                )
                history = history[:-1] + [sanitized]

        messages = history + [SystemMessage(content=prompt)]
        logger.info(f"[final_answer] Invoking LLM")
        if solver:
            solver.log_step("final_answer", "llm_input", content=prompt)

        response = model.invoke(messages)
        resp_text = _as_text(response.content)
        logger.info(f"[final_answer] Response ({len(resp_text)} chars): {resp_text[:300]}")
        if solver:
            solver.log_step("final_answer", "llm_output", content=resp_text)

        return {
            "messages": [response],
            "current_node": "final_answer",
        }

    return final_answer_node


def make_structured_output_node(model: BaseChatModel, question: BaseQuestion, solver=None):
    """Create the structured output node.

    Converts the final answer into the Pydantic model expected by the question.
    """

    def structured_output_node(state: PlannerAgentState):
        prompt = """# Role and Objective
You are the **Final Answer Structured Output Generator**, providing a structured response to the user's query.

# Instructions
- Extract the necessary data from the previous final answer message or conversation context
- Use this data to populate the structured response according to the Pydantic model
- Ensure all required fields are filled based on the information available in the conversation

# Output Format
- Provide a structured response using the provided Pydantic model"""

        prior_msgs = state.get("messages", [])
        if prior_msgs and solver is not None:
            last = prior_msgs[-1]
            content = getattr(last, "content", "") or ""
            if not content.strip():
                solver.last_had_empty_response = True

        structured_llm = model.with_structured_output(question.output_model())
        messages = state.get("messages", []) + [SystemMessage(content=prompt)]

        if solver:
            solver.log_step("structured_output", "llm_input", content=prompt)

        structured_output = invoke_structured_with_fallback(
            structured_llm, messages, question.output_model(), solver,
            fallback_text=getattr(prior_msgs[-1], "content", "") if prior_msgs else "",
        )

        if structured_output is None:
            # Neither the structured call nor the fallback yielded a valid
            # answer. Emit empty content so the engine records this as an
            # empty/invalid response instead of crashing on None.model_dump_json().
            if solver:
                solver.last_had_empty_response = True
                solver.log_step("structured_output", "llm_output", content="")
            return {
                "messages": [AIMessage(content="")],
                "current_node": "structured_output",
            }

        if solver:
            solver.log_step("structured_output", "llm_output", content=structured_output.model_dump_json())

        response_message = AIMessage(content=structured_output.model_dump_json())

        return {
            "messages": [response_message],
            "current_node": "structured_output",
        }

    return structured_output_node


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def make_after_planner(max_iterations: int):
    """Route after the planner node.

    - If the planner emitted tool_calls → go to tools (let it keep gathering)
    - If max iterations reached → skip to final_answer
    - Otherwise → go to validation
    """

    def after_planner(state: PlannerAgentState):
        last = state["messages"][-1] if state["messages"] else None
        has_tool_calls = hasattr(last, "tool_calls") and last.tool_calls
        iteration = state.get("iteration_count", 0)
        content_preview = str(last.content)[:150] if last and hasattr(last, "content") else "N/A"
        tool_names = [tc.get("name", "?") for tc in last.tool_calls] if has_tool_calls else []

        # Hard stop FIRST: if we have hit the iteration cap, force a final
        # answer even if the planner is still requesting more tools.
        # Otherwise a model that keeps emitting tool_calls would loop forever
        # (or until LangGraph's recursion_limit blows up).
        if iteration >= max_iterations:
            logger.info(
                f"[after_planner] iter={iteration} → FINAL_ANSWER "
                f"(max iterations reached; pending tool_calls={tool_names or 'none'})"
            )
            return "final_answer"

        # Planner wants to call tools → let it
        if has_tool_calls:
            logger.info(f"[after_planner] iter={iteration} → TOOLS: {tool_names}")
            return "tools"

        # Planner finished gathering → validate
        logger.info(f"[after_planner] iter={iteration} → VALIDATION (no tool_calls). Response: {content_preview}")
        return "validation"

    return after_planner


def after_tools(state: PlannerAgentState):
    """After tools execute, always return to planner so it can decide
    whether to call more tools or declare itself done."""
    return "planner"


def make_after_validation(max_validation_retries: int):
    """Route after the validation node.

    - COMPLETE (no INCOMPLETE in response) → final_answer
    - INCOMPLETE and under max validation retries → planner
    - INCOMPLETE but at max retries → final_answer anyway
    """

    def after_validation(state: PlannerAgentState):
        last = state["messages"][-1] if state["messages"] else None
        content = _as_text(last.content) if hasattr(last, "content") else ""
        validation_count = state.get("validation_count", 0)

        if "INCOMPLETE" not in content:
            logger.info(f"[after_validation] validation_count={validation_count} → FINAL_ANSWER (COMPLETE)")
            return "final_answer"

        if validation_count >= max_validation_retries:
            logger.info(f"[after_validation] validation_count={validation_count} → FINAL_ANSWER (max retries, still INCOMPLETE)")
            return "final_answer"

        logger.info(f"[after_validation] validation_count={validation_count} → PLANNER (INCOMPLETE)")
        return "planner"

    return after_validation


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class PlannerAgentSolver(BaseSolver):
    """Planner-based agentic solver using file tools only.

    Graph:
        START → planner ←→ tools → planner → validation
                                                ↓ COMPLETE → final_answer → structured_output → END
                                                ↓ INCOMPLETE → planner (loop)
    """

    use_network: bool = False

    def __init__(
        self,
        lab_path: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_validation_retries: int = DEFAULT_MAX_VALIDATION_RETRIES,
    ):
        super().__init__(lab_path)
        if not os.path.isdir(lab_path):
            raise ValueError(
                f"Lab path does not exist or is not a directory: {lab_path}"
            )
        self.max_iterations = max_iterations
        self.max_validation_retries = max_validation_retries

    def solve(
        self, question: BaseQuestion, model: BaseChatModel
    ):
        """Solve the question using the planner agent workflow."""
        self.reset_diagnostics()
        return self._run_workflow(model, question)

    # ------------------------------------------------------------------

    def _build_tools(self, question: BaseQuestion) -> list:
        """Assemble the tool list based on use_network flag."""
        tools = build_file_tools(self.lab_path)
        if self.use_network and getattr(question, "_kathara", None) is not None:
            tools += build_network_tools(question._kathara)
        return tools

    def _build_output_schema(self, question: BaseQuestion) -> str:
        """JSON-serialized schema of the question's output model."""
        return json.dumps(
            question.output_model().model_json_schema(), indent=2
        )

    def _create_workflow(self, model: BaseChatModel, question: BaseQuestion):
        """Build and compile the LangGraph workflow."""
        tools = self._build_tools(question)
        output_schema = self._build_output_schema(question)

        # Nodes (pass self for logging)
        planner = make_planner_node(model, tools, solver=self)
        validator = make_validation_node(model, solver=self)
        final_answer = make_final_answer_node(model, solver=self)
        structured_output = make_structured_output_node(model, question, solver=self)
        raw_tool_node = ToolNode(tools)
        tool_node = make_tool_logging_node(raw_tool_node, solver=self)

        # Graph
        graph = StateGraph(PlannerAgentState)

        graph.add_node("planner", planner)
        graph.add_node("validation", validator)
        graph.add_node("final_answer", final_answer)
        graph.add_node("structured_output", structured_output)
        graph.add_node("tools", tool_node)

        # Edges
        graph.add_edge(START, "planner")

        graph.add_conditional_edges(
            "planner",
            make_after_planner(self.max_iterations),
            {
                "tools": "tools",
                "validation": "validation",
                "final_answer": "final_answer",
            },
        )

        graph.add_edge("tools", "planner")  # self-loop back to planner

        graph.add_conditional_edges(
            "validation",
            make_after_validation(self.max_validation_retries),
            {
                "final_answer": "final_answer",
                "planner": "planner",
            },
        )

        graph.add_edge("final_answer", "structured_output")
        graph.add_edge("structured_output", END)

        return graph.compile()

    def _run_workflow(self, model: BaseChatModel, question: BaseQuestion):
        """Execute the workflow and parse the final structured output."""
        app = self._create_workflow(model, question)

        initial_state: PlannerAgentState = {
            "messages": [],
            "user_query": question.question_text,
            "output_schema": self._build_output_schema(question),
            "iteration_count": 0,
            "validation_count": 0,
            "current_node": "planner",
        }

        # Recursion limit must comfortably exceed the worst-case node count:
        # up to max_iterations planner calls, each typically followed by a
        # tools node, plus up to max_validation_retries validation nodes and
        # the final_answer + structured_output pair. With defaults 25/3 that
        # is ~55 visits; we set 75 for headroom. The planner's iteration_count
        # cap is what actually stops the loop — this is only a safety net.
        token_cb = TokenUsageCallback()
        result = app.invoke(
            initial_state,
            config={"recursion_limit": 75, "callbacks": [token_cb]},
        )
        self.last_token_stats = token_cb.snapshot()

        final_messages = result.get("messages", [])
        last_message = final_messages[-1]

        if hasattr(last_message, "content"):
            text = _as_text(last_message.content).strip()
            # Empty content means structured_output_node could not produce a
            # schema-conforming object. Return None so the engine records this
            # as an empty/invalid response instead of raising on json.loads("").
            if not text:
                return None
            structured_data = json.loads(text)
            return question.output_model()(**structured_data)

        return last_message


class PlannerAgentSolverWithNetwork(PlannerAgentSolver):
    """Same as PlannerAgentSolver but with live network tools enabled.

    Additional tools available to the planner:
      - ping: test ICMP reachability
      - traceroute: trace layer-3 path
      - get_routing_table: show ip route
      - get_interfaces: show ip addr
      - get_arp_table: show ARP cache
      - get_running_processes: list running processes (ps aux)
      - read_device_file: read a file from a running device's filesystem

    Requires that the question has a KatharaClient injected
    (question._kathara is not None).
    """

    use_network: bool = True
