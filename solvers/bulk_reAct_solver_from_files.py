import json
import os
import logging
from typing import Annotated, Optional, TypedDict
from .base_solver import BaseSolver, invoke_structured_with_fallback
from langchain.chat_models.base import BaseChatModel
from questions.base_question import BaseQuestion

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from core.token_callback import TokenUsageCallback

logger = logging.getLogger(__name__)

# Define the state
class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]

# Node factories

def make_reason_node(model: BaseChatModel, question: BaseQuestion, lab_context: str, solver=None):
    def reason_node(state: WorkflowState):
        system_prompt = """# Role
You are a Network Configuration Analyst. Your goal is to analyze the provided Kathara lab files and plan how to answer the question.

# Instructions
1. Analyze the network topology from the configuration files.
2. Identify which devices and files are relevant.
3. Extract specific details (IPs, routes, etc.).
4. Formulate a step-by-step logic to answer the question.

Think step-by-step. Be concise but thorough."""

        human_prompt = f"""# Lab Configuration Files
{lab_context}

# User Question
{question.question_text}"""

        messages = state.get("messages", []) + [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        logger.info(f"[BulkReact:reason] Invoking LLM (context_len={len(lab_context)})")

        if solver:
            solver.log_step("reason", "llm_input", content=human_prompt[:500], truncated=True)

        response = model.invoke(messages)
        logger.info(f"[BulkReact:reason] LLM response ({len(response.content)} chars)")

        if solver:
            solver.log_step("reason", "llm_output", content=response.content)

        return {"messages": [response]}
    return reason_node

def make_act_node(model: BaseChatModel, question: BaseQuestion, solver=None):
    def act_node(state: WorkflowState):
        system_prompt = """# Role
You are the Action Phase agent. Based on the previous reasoning, provide a final, concrete answer.

# Instructions
1. Review the reasoning provided in the previous step.
2. Synthesize a direct answer to the question.
3. Include specific names, IPs, and values discovered.
4. If the answer cannot be determined, state why clearly.

Provide a clear and definitive final answer."""

        human_prompt = f"""# User Question
{question.question_text}

Based on the reasoning above, give the final answer now."""

        messages = state.get("messages", []) + [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        logger.info("[BulkReact:act] Invoking LLM")

        if solver:
            solver.log_step("act", "llm_input", content=human_prompt)

        response = model.invoke(messages)
        logger.info(f"[BulkReact:act] LLM response ({len(response.content)} chars)")

        if solver:
            solver.log_step("act", "llm_output", content=response.content)

        return {"messages": [response]}
    return act_node

def make_structured_output_node(model: BaseChatModel, question: BaseQuestion, solver=None):
    def structured_output_node(state: WorkflowState):
        system_prompt = """# Role
Generate a structured JSON response based on the conversation history.

# Instructions
- Populate the fields of the Pydantic model exactly.
- Use only information gathered in the previous steps."""

        human_prompt = "Produce the structured JSON answer now based on the conversation above."

        prior_msgs = state.get("messages", [])
        if prior_msgs and solver is not None:
            last = prior_msgs[-1]
            content = getattr(last, "content", "") or ""
            if not content.strip():
                solver.last_had_empty_response = True

        structured_llm = model.with_structured_output(question.output_model())
        messages = state.get("messages", []) + [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        logger.info("[BulkReact:structured_output] Invoking structured LLM")
        if solver:
            solver.log_step("structured_output", "llm_input", content=human_prompt)

        structured_output = invoke_structured_with_fallback(
            structured_llm, messages, question.output_model(), solver,
            fallback_text=getattr(prior_msgs[-1], "content", "") if prior_msgs else "",
        )

        if structured_output is None:
            if solver:
                solver.last_had_empty_response = True
                solver.log_step("structured_output", "llm_output", content="")
            return {"messages": [AIMessage(content="")]}

        if solver:
            solver.log_step("structured_output", "llm_output", content=structured_output.model_dump_json())

        response_message = AIMessage(content=structured_output.model_dump_json())
        return {"messages": [response_message]}
    return structured_output_node

# Graph factory

def create_workflow(model: BaseChatModel, question: BaseQuestion, lab_context: str, solver=None):
    """Create the LangGraph workflow with all nodes and edges."""

    # Create nodes
    reason_node = make_reason_node(model, question, lab_context, solver=solver)
    act_node = make_act_node(model, question, solver=solver)
    structured_output_node = make_structured_output_node(model, question, solver=solver)
    
    # Build the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("reason", reason_node)
    workflow.add_node("act", act_node)
    workflow.add_node("structured_output", structured_output_node)
    
    # Add edges
    workflow.add_edge(START, "reason")
    workflow.add_edge("reason", "act")
    workflow.add_edge("act", "structured_output")
    workflow.add_edge("structured_output", END)

    return workflow.compile()


class BulkReactSolverFromFiles(BaseSolver):
    """
    Prompt builder that gathers all .conf and .startup files from the Kathara lab directory
    and attaches their content as context to the question.
    """

    def __init__(self, lab_path: str):
        """
        Initializes the prompt builder with the lab path.
        
        Args:
            lab_path (str): Path to the Kathara lab folder
        """
        super().__init__(lab_path)
        if not os.path.isdir(lab_path):
            raise ValueError(f"The provided lab path does not exist or is not a directory: {lab_path}")
        
    def solve(self, question: BaseQuestion, model: BaseChatModel):
        """
        Solves the question using the provided model and the gathered context from files.

        Args:
            question (BaseQuestion): The question to solve.
            model (BaseChatModel): The chat model to use for solving the question.

        Returns:
            BaseModel | None: An instance of the question's output model, or None on failure.
        """
        self.reset_diagnostics()
        # Build the prompt with context from files

        lab_context = self._gather_context_files()

        answer = self.run_workflow(model, question, lab_context)
        return answer

    def _gather_context_files(self) -> str:
        """
        Traverse the lab folder and retrieve contents of .conf and .startup files.
        
        Returns:
            str: Combined context from the relevant files, with headers showing the lab folder
                 base name and the file's relative path (including the file itself).
        """
        context_parts = []
        lab_base = os.path.basename(os.path.normpath(self.lab_path))
        for root, _, files in os.walk(self.lab_path):
            for filename in files:
                if filename.endswith('.conf') or filename.endswith('.startup') or filename.endswith('.txt'):
                    file_path = os.path.join(root, filename)
                    # Generate a relative path and attach the lab folder base at the beginning
                    rel_path = os.path.relpath(file_path, self.lab_path).replace(os.sep, '/')
                    full_rel_path = f"{lab_base}/{rel_path}"
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        context_parts.append(f"**PATH:** {full_rel_path} \n```\n{file_content}\n```\n\n")
                    except Exception as e:
                        logger.error(f"Error reading file {full_rel_path}: {e}")
        
        if not context_parts:
            logger.warning(f"No .conf or .startup files found in {self.lab_path}")
            
        return "\n\n".join(context_parts)

    # def _build_prompt(self, question_text: str) -> str:
    #     """
    #     Constructs the full prompt with lab context and the provided question text.
        
    #     Args:
    #         question_text (str): The question from the question plugin
        
    #     Returns:
    #         str: The complete prompt ready for the LLM
    #     """
    #     context = self._gather_context_files()
    #     complete_prompt = f"*Lab Context:*\n{context}\n\nYou will only have access to the network configuration above and won't have access to the network itself to answer the user's question.\n\n*Question:*\n{question_text}"
    #     return complete_prompt

    def run_workflow(self, model: BaseChatModel, question: BaseQuestion, lab_context: str):
        """Runs the LangGraph workflow to solve the question using the provided model."""

        # Create the workflow
        app = create_workflow(model, question, lab_context, solver=self)

        # Initial state
        initial_state = {
            "messages": []
        }

        # Run the workflow with a token-usage callback.
        token_cb = TokenUsageCallback()
        result = app.invoke(initial_state, config={"callbacks": [token_cb]})
        self.last_token_stats = token_cb.snapshot()

        # Extract final answer
        final_messages = result.get("messages", [])
        last_message = final_messages[-1]

        # Parse the JSON content back to Pydantic model. Empty content means the
        # structured_output node could not produce (or recover) a valid answer;
        # return None so the engine records an empty/invalid response instead of
        # raising on json.loads("").
        if hasattr(last_message, 'content'):
            content = last_message.content
            if not (content.strip() if isinstance(content, str) else content):
                return None
            structured_data = json.loads(content)
            return question.output_model()(**structured_data)

        return final_messages[-1]