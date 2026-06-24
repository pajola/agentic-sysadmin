"""
Export LangGraph workflow graphs as PNG and PDF images.

Usage:
    python export_graphs.py

Outputs (each as both .png and .pdf):
    assets/graph_bulk_processing
    assets/graph_bulk_react
    assets/graph_planner_agent
    assets/graph_planner_agent_with_network
    assets/graph_guided_retrieval_agent
"""

import base64
import os
import time
from typing import Annotated, Dict, List, Any, TypedDict

import requests
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --------------------------------------------------------------------------- #
# State definitions
# --------------------------------------------------------------------------- #

class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]


class PlannerState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    output_schema: str
    iteration_count: int
    current_node: str


class GuidedRetrievalState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    output_schema: str
    retrieval_strategy: str
    assembled_context: str
    probe_results: str
    current_node: str


# --------------------------------------------------------------------------- #
# Stub tools
# --------------------------------------------------------------------------- #

@tool
def list_lab_files() -> Dict[str, Any]:
    """List all files in the Kathara lab directory."""
    return {"status": "success", "topology": [], "startup": [], "config": []}

@tool
def read_lab_conf() -> Dict[str, Any]:
    """Read the lab.conf topology file."""
    return {"status": "success", "content": ""}

@tool
def read_file(relative_path: str) -> Dict[str, Any]:
    """Read a specific file from the lab directory."""
    return {"status": "success", "path": relative_path, "content": ""}

@tool
def get_devices_name() -> Dict[str, List[str]]:
    """Retrieves all device names in the network."""
    return {"status": "success", "devices": []}

@tool
def get_device_config(device_name: str) -> Dict[str, Any]:
    """Retrieves the configuration for a given device name."""
    return {"status": "success", "device_name": device_name, "device_config": {}}

@tool
def execute_command(device_name: str, command: str) -> Dict[str, Any]:
    """Execute a command on a specific device in the network."""
    return {"status": "success", "device_name": device_name, "command": command, "output": "", "error": None}

@tool
def ping(source_device: str, destination_ip: str, count: int = 3) -> Dict[str, Any]:
    """Test network reachability using ICMP ping."""
    return {"status": "success", "source": source_device, "destination": destination_ip, "reachable": True, "output": ""}

@tool
def traceroute(source_device: str, destination_ip: str) -> Dict[str, Any]:
    """Trace the network path from a device to a destination."""
    return {"status": "success", "source": source_device, "destination": destination_ip, "output": ""}

@tool
def get_routing_table(device_name: str) -> Dict[str, Any]:
    """Get the full IP routing table of a network device."""
    return {"status": "success", "device_name": device_name, "output": ""}

@tool
def get_interfaces(device_name: str) -> Dict[str, Any]:
    """Get all network interfaces and their IP addresses."""
    return {"status": "success", "device_name": device_name, "output": ""}

@tool
def get_arp_table(device_name: str) -> Dict[str, Any]:
    """Get the ARP table of a device."""
    return {"status": "success", "device_name": device_name, "output": ""}


# --------------------------------------------------------------------------- #
# Stub node helpers
# --------------------------------------------------------------------------- #

def stub_node(name: str):
    def node(state):
        return {"messages": [AIMessage(content=f"stub:{name}")]}
    node.__name__ = name
    return node


def stub_planner_node(name: str):
    def node(state):
        return {
            "messages": [AIMessage(content=f"stub:{name}")],
            "current_node": name,
        }
    node.__name__ = name
    return node


# --------------------------------------------------------------------------- #
# Bulk Processing graph: llm_solver -> structured_output
# --------------------------------------------------------------------------- #

def build_bulk_processing_graph():
    workflow = StateGraph(SimpleState)
    workflow.add_node("llm_solver", stub_node("llm_solver"))
    workflow.add_node("structured_output", stub_node("structured_output"))

    workflow.add_edge(START, "llm_solver")
    workflow.add_edge("llm_solver", "structured_output")
    workflow.add_edge("structured_output", END)

    return workflow.compile()


# --------------------------------------------------------------------------- #
# Bulk ReAct graph: reason -> act -> structured_output
# --------------------------------------------------------------------------- #

def build_bulk_react_graph():
    workflow = StateGraph(SimpleState)
    workflow.add_node("reason", stub_node("reason"))
    workflow.add_node("act", stub_node("act"))
    workflow.add_node("structured_output", stub_node("structured_output"))

    workflow.add_edge(START, "reason")
    workflow.add_edge("reason", "act")
    workflow.add_edge("act", "structured_output")
    workflow.add_edge("structured_output", END)

    return workflow.compile()


# --------------------------------------------------------------------------- #
# Planner Agent graph
# --------------------------------------------------------------------------- #

def _planner_decide_next(state):
    last = state["messages"][-1] if state["messages"] else None
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "validation"


def _validation_decide_next(state):
    last = state["messages"][-1] if state["messages"] else None
    content = last.content if hasattr(last, "content") else ""
    if "INCOMPLETE" in content:
        return "planner"
    return "final_answer"


def build_planner_agent_graph(include_network_tools: bool = False):
    file_tools = [list_lab_files, read_lab_conf, read_file, get_devices_name, get_device_config]
    if include_network_tools:
        file_tools += [execute_command, ping, traceroute, get_routing_table, get_interfaces, get_arp_table]

    tool_node = ToolNode(file_tools)

    workflow = StateGraph(PlannerState)
    workflow.add_node("planner", stub_planner_node("planner"))
    workflow.add_node("tools", tool_node)
    workflow.add_node("validation", stub_planner_node("validation"))
    workflow.add_node("final_answer", stub_planner_node("final_answer"))
    workflow.add_node("structured_output", stub_planner_node("structured_output"))

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        _planner_decide_next,
        {"tools": "tools", "validation": "validation"},
    )
    workflow.add_edge("tools", "planner")
    workflow.add_conditional_edges(
        "validation",
        _validation_decide_next,
        {"planner": "planner", "final_answer": "final_answer"},
    )
    workflow.add_edge("final_answer", "structured_output")
    workflow.add_edge("structured_output", END)

    return workflow.compile()


# --------------------------------------------------------------------------- #
# Guided Retrieval Agent graph (formerly StrategicAgentSolver)
#
#   START → strategy_classifier → context_assembler → analyst
#         → structured_output → END
# --------------------------------------------------------------------------- #

def build_guided_retrieval_agent_graph():
    workflow = StateGraph(GuidedRetrievalState)
    workflow.add_node("strategy_classifier", stub_planner_node("strategy_classifier"))
    workflow.add_node("context_assembler", stub_planner_node("context_assembler"))
    workflow.add_node("analyst", stub_planner_node("analyst"))
    workflow.add_node("structured_output", stub_planner_node("structured_output"))

    workflow.add_edge(START, "strategy_classifier")
    workflow.add_edge("strategy_classifier", "context_assembler")
    workflow.add_edge("context_assembler", "analyst")
    workflow.add_edge("analyst", "structured_output")
    workflow.add_edge("structured_output", END)

    return workflow.compile()


# --------------------------------------------------------------------------- #
# Export
#
# Graphs are rendered through the public mermaid.ink service:
#   - PNG: /img endpoint, rasterised at high resolution (width x scale).
#   - PDF: /pdf endpoint, a TRUE VECTOR document (embedded fonts, no bitmap),
#          so it stays crisp at any zoom level. `fit=true` crops the page to
#          the diagram instead of dropping it onto an A4 sheet.
#
# The previous approach embedded the small mermaid PNG into a PDF with PIL,
# which produced a blurry, pixelated raster PDF regardless of the DPI metadata.
# --------------------------------------------------------------------------- #

MERMAID_INK = "https://mermaid.ink"
# mermaid.ink rejects requests without a browser-like User-Agent (HTTP 403).
_HEADERS = {"User-Agent": "Mozilla/5.0 (export_graphs.py)"}

# PNG resolution. mermaid.ink only honours `scale` when `width` is also set,
# so the rendered image is roughly PNG_WIDTH * PNG_SCALE pixels wide.
PNG_WIDTH = 1600
PNG_SCALE = 2


def _fetch_mermaid_ink(
    mermaid_syntax: str,
    path: str,
    query: str,
    *,
    max_retries: int = 5,
    retry_delay: float = 3.0,
) -> bytes:
    """GET an image/PDF from mermaid.ink, retrying on transient errors."""
    encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode("ascii")
    url = f"{MERMAID_INK}/{path}/{encoded}?{query}"

    for attempt in range(max_retries + 1):
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 200:
            return resp.content
        # Retry rate-limit / server errors with linear backoff.
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            time.sleep(retry_delay * (attempt + 1))
            continue
        raise RuntimeError(
            f"mermaid.ink /{path} request failed (HTTP {resp.status_code})"
        )
    raise RuntimeError(
        f"mermaid.ink /{path} request failed after {max_retries} retries"
    )


def export_graph(app, output_stem: str, *, bg_color: str = "white"):
    """Render the graph as a high-resolution PNG and a vector PDF."""
    os.makedirs(os.path.dirname(output_stem) or ".", exist_ok=True)
    mermaid_syntax = app.get_graph().draw_mermaid()

    png_bytes = _fetch_mermaid_ink(
        mermaid_syntax,
        "img",
        f"type=png&bgColor={bg_color}&width={PNG_WIDTH}&scale={PNG_SCALE}",
    )
    png_path = f"{output_stem}.png"
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    print(f"Saved: {png_path}")

    pdf_bytes = _fetch_mermaid_ink(
        mermaid_syntax,
        "pdf",
        f"bgColor={bg_color}&fit=true",
    )
    pdf_path = f"{output_stem}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"Saved: {pdf_path} (vector)")


if __name__ == "__main__":
    graphs = {
        "assets/graph_bulk_processing": build_bulk_processing_graph,
        "assets/graph_bulk_react": build_bulk_react_graph,
        "assets/graph_planner_agent": lambda: build_planner_agent_graph(False),
        "assets/graph_planner_agent_with_network": lambda: build_planner_agent_graph(True),
        "assets/graph_guided_retrieval_agent": build_guided_retrieval_agent_graph,
    }

    for stem, builder in graphs.items():
        app = builder()
        export_graph(app, stem)
