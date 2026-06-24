"""
Strategic Agent Solver — question-routing with deterministic context curation.

Architecture:
    START → strategy_classifier (1 LLM call — picks retrieval strategy)
          → context_assembler   (deterministic — reads the right files)
          → prober              (deterministic — live network commands, if needed)
          → analyst             (1 LLM call — reasons over the curated context)
          → structured_output   (1 LLM call — extracts the Pydantic answer)
          → END

Key design principles:
  - The LLM classifies the question into a retrieval strategy (agentic decision)
  - The system gathers data deterministically based on the chosen strategy
  - The analyst receives curated, focused context and reasons about it
  - No tools are bound to the analyst: the context is already curated, and
    binding tools confused small models (they would either return empty content
    or hallucinate tool names instead of using the provided context).
  - Total: 3 LLM calls (vs 4-10 for PlannerAgentSolver, vs 2-3 for bulk)
"""

import json
import logging
import os
from typing import TypedDict, Annotated, Literal

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from questions.base_question import BaseQuestion
from core.token_callback import TokenUsageCallback
from .base_solver import BaseSolver, invoke_structured_with_fallback
from .context_strategies import CONTEXT_ASSEMBLERS, PROBERS
from .prompt_templates import CLASSIFIER_PROMPT, ANALYST_PROMPTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model for the classifier's structured output
# ---------------------------------------------------------------------------

class StrategyChoice(BaseModel):
    """The retrieval strategy chosen by the classifier."""
    strategy: Literal[
        "topology_only", "ip_analysis", "device_pair",
        "live_connectivity", "service_scan"
    ] = Field(description="The chosen retrieval strategy")
    reasoning: str = Field(description="Brief explanation for the choice")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class GuidedRetrievalAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    output_schema: str
    retrieval_strategy: str
    assembled_context: str
    probe_results: str
    current_node: str


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_strategy_classifier_node(model: BaseChatModel, solver=None):
    """LLM classifies the question into one of 5 retrieval strategies."""

    def strategy_classifier_node(state: GuidedRetrievalAgentState):
        prompt = CLASSIFIER_PROMPT.format(
            user_query=state["user_query"],
            output_schema=state["output_schema"],
        )

        logger.info("[strategy_classifier] Invoking LLM for classification")
        if solver:
            solver.log_step("strategy_classifier", "llm_input", content=prompt)

        structured_llm = model.with_structured_output(StrategyChoice)
        result = structured_llm.invoke([SystemMessage(content=prompt)])

        strategy = result.strategy
        reasoning = result.reasoning

        logger.info(f"[strategy_classifier] Chose strategy: {strategy} — {reasoning}")
        if solver:
            solver.log_step("strategy_classifier", "llm_output",
                            content=f"{strategy}: {reasoning}",
                            strategy=strategy, reasoning=reasoning)

        return {
            "retrieval_strategy": strategy,
            "current_node": "strategy_classifier",
        }

    return strategy_classifier_node


def make_context_assembler_node(lab_path: str, question: BaseQuestion, solver=None):
    """Deterministically reads the right files based on the chosen strategy."""

    def context_assembler_node(state: GuidedRetrievalAgentState):
        strategy = state["retrieval_strategy"]
        assembler = CONTEXT_ASSEMBLERS.get(strategy, CONTEXT_ASSEMBLERS["ip_analysis"])

        logger.info(f"[context_assembler] Assembling context for strategy: {strategy}")
        context = assembler(lab_path, question=question)

        if solver:
            solver.log_step("context_assembler", "context_assembly",
                            strategy=strategy,
                            context_length=len(context))

        return {
            "assembled_context": context,
            "current_node": "context_assembler",
        }

    return context_assembler_node


def make_prober_node(question: BaseQuestion, lab_path: str, solver=None):
    """Deterministically executes network probes if the strategy requires it."""

    def prober_node(state: GuidedRetrievalAgentState):
        strategy = state["retrieval_strategy"]
        prober = PROBERS.get(strategy, PROBERS["topology_only"])

        logger.info(f"[prober] Running probe for strategy: {strategy}")
        results = prober(question, lab_path)

        if solver:
            solver.log_step("prober", "probe_execution",
                            strategy=strategy,
                            results_length=len(results))

        return {
            "probe_results": results,
            "current_node": "prober",
        }

    return prober_node


def make_analyst_node(model: BaseChatModel, solver=None):
    """The analyst reasons over the curated context. No tools — the context
    has already been assembled deterministically by the upstream nodes."""

    def analyst_node(state: GuidedRetrievalAgentState):
        strategy = state["retrieval_strategy"]

        template = ANALYST_PROMPTS.get(strategy, ANALYST_PROMPTS["ip_analysis"])
        prompt = template.format(
            user_query=state["user_query"],
            output_schema=state["output_schema"],
            assembled_context=state.get("assembled_context", ""),
            probe_results=state.get("probe_results", ""),
        )

        logger.info(f"[analyst] Invoking LLM (strategy={strategy})")
        if solver:
            solver.log_step("analyst", "llm_input", content=prompt[:500])

        response = model.invoke([SystemMessage(content=prompt)])

        logger.info(f"[analyst] Response length: {len(response.content or '')}")
        if solver:
            solver.log_step("analyst", "llm_output", content=response.content)

        return {
            "messages": [response],
            "current_node": "analyst",
        }

    return analyst_node


def make_structured_output_node(model: BaseChatModel, question: BaseQuestion, solver=None):
    """Convert the analyst's answer into the Pydantic output model.

    If the analyst returned empty content (rare without tools, but possible),
    re-include the assembled context so the model can still produce a grounded
    answer instead of hallucinating from nothing.
    """

    def structured_output_node(state: GuidedRetrievalAgentState):
        # Grab the analyst's text response (last message)
        msgs = state.get("messages", [])
        analyst_text = ""
        if msgs:
            last = msgs[-1]
            analyst_text = getattr(last, "content", "") or ""

        if analyst_text.strip():
            prompt = (
                "Extract the answer from the analysis below and produce a "
                "structured JSON response that satisfies the provided Pydantic "
                "model. Populate every field using only the information given.\n\n"
                f"# Analysis\n{analyst_text}"
            )
        else:
            # Fallback: analyst returned empty. Re-include curated context so
            # the structured_output node still has something to work with.
            logger.warning("[structured_output] Analyst returned empty content; "
                           "falling back to assembled context.")
            if solver is not None:
                solver.last_had_empty_response = True
            prompt = (
                "Produce a structured JSON response that satisfies the provided "
                "Pydantic model, using only the information in the lab "
                "configuration and probe results below.\n\n"
                f"# Question\n{state['user_query']}\n\n"
                f"# Lab Configuration\n{state.get('assembled_context', '')}\n\n"
                f"# Probe Results\n{state.get('probe_results', '')}"
            )

        structured_llm = model.with_structured_output(question.output_model())

        logger.info("[structured_output] Invoking LLM")
        if solver:
            solver.log_step("structured_output", "llm_input", content=prompt[:500])

        # Fall back to the analyst's prose (which usually already contains the
        # answer as a JSON block) if the structured call returns None / raises.
        structured_output = invoke_structured_with_fallback(
            structured_llm, [SystemMessage(content=prompt)], question.output_model(),
            solver, fallback_text=analyst_text,
        )

        if structured_output is None:
            if solver:
                solver.last_had_empty_response = True
                solver.log_step("structured_output", "llm_output", content="")
            return {
                "messages": [AIMessage(content="")],
                "current_node": "structured_output",
            }

        if solver:
            solver.log_step("structured_output", "llm_output",
                            content=structured_output.model_dump_json())

        response_message = AIMessage(content=structured_output.model_dump_json())
        return {
            "messages": [response_message],
            "current_node": "structured_output",
        }

    return structured_output_node


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_context(state: GuidedRetrievalAgentState):
    """After context assembly: go to prober if live strategy, else analyst."""
    strategy = state["retrieval_strategy"]
    if strategy in ("live_connectivity", "service_scan"):
        return "prober"
    return "analyst"


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class GuidedRetrievalAgentSolver(BaseSolver):
    """Strategic agent solver using question-routing and deterministic context curation.

    Graph:
        START → strategy_classifier → context_assembler → [prober] → analyst
              → structured_output → END

    Live network probing (traceroute, ps aux, etc.) is performed deterministically
    by the prober node for the `live_connectivity` and `service_scan` strategies,
    using the question's Kathara client when available.
    """

    def __init__(self, lab_path: str):
        super().__init__(lab_path)
        if not os.path.isdir(lab_path):
            raise ValueError(f"Lab path does not exist or is not a directory: {lab_path}")

    def solve(self, question: BaseQuestion, model: BaseChatModel):
        self.reset_diagnostics()
        app = self._create_workflow(model, question)

        initial_state: GuidedRetrievalAgentState = {
            "messages": [],
            "user_query": question.question_text,
            "output_schema": json.dumps(
                question.output_model().model_json_schema(), indent=2
            ),
            "retrieval_strategy": "",
            "assembled_context": "",
            "probe_results": "",
            "current_node": "strategy_classifier",
        }

        token_cb = TokenUsageCallback()
        result = app.invoke(
            initial_state,
            config={"recursion_limit": 10, "callbacks": [token_cb]},
        )
        self.last_token_stats = token_cb.snapshot()

        final_messages = result.get("messages", [])
        last_message = final_messages[-1]

        if hasattr(last_message, "content"):
            content = last_message.content
            # Empty content means structured_output could not produce (or
            # recover) a valid answer; return None so the engine records an
            # empty/invalid response instead of raising on json.loads("").
            if not (content.strip() if isinstance(content, str) else content):
                return None
            structured_data = json.loads(content)
            return question.output_model()(**structured_data)
        return last_message

    def _create_workflow(self, model: BaseChatModel, question: BaseQuestion):
        # Nodes
        classifier = make_strategy_classifier_node(model, solver=self)
        assembler = make_context_assembler_node(self.lab_path, question, solver=self)
        prober = make_prober_node(question, self.lab_path, solver=self)
        analyst = make_analyst_node(model, solver=self)
        structured_output = make_structured_output_node(model, question, solver=self)

        # Graph
        graph = StateGraph(GuidedRetrievalAgentState)

        graph.add_node("strategy_classifier", classifier)
        graph.add_node("context_assembler", assembler)
        graph.add_node("prober", prober)
        graph.add_node("analyst", analyst)
        graph.add_node("structured_output", structured_output)

        # Edges
        graph.add_edge(START, "strategy_classifier")
        graph.add_edge("strategy_classifier", "context_assembler")

        graph.add_conditional_edges(
            "context_assembler",
            route_after_context,
            {"prober": "prober", "analyst": "analyst"},
        )

        graph.add_edge("prober", "analyst")
        graph.add_edge("analyst", "structured_output")
        graph.add_edge("structured_output", END)

        return graph.compile()
