"""
MulBrandEval — LangGraph DAG Pipeline
Phase 2A: English Baseline

DAG execution order:
  Node 0 (Prompt Parser)
    → Node 1 (Object Presence)
        → [FAIL short-circuit] → Node 6 (Coordinator)
        → [PASS] → Node 2 (Spatial Layout)
                     → Node 3 (Colour Attribute)
                         → Node 4 (Typography)
                             → Node 5 (Brand Quality)
                                 → Node 6 (Coordinator)
                                     → END

Short-circuit rationale: if required objects are absent, Nodes 2–5
are meaningless to run. Spatial/colour checks on absent objects
produce noise, not diagnostic information. Node 6 scores all
short-circuited nodes as 0.0.

Nodes 2, 3, 4, 5 are logically independent of each other but
executed sequentially — functionally equivalent to parallel since
none depend on each other's output. LangGraph sequential edges
are used for simplicity and debuggability.
"""
from langgraph.graph import StateGraph, END
from phase2a_dag_pipeline.state import BrandComplianceState
from phase2a_dag_pipeline.nodes.node0_prompt_parser  import run_node0
from phase2a_dag_pipeline.nodes.node1_object_presence import run_node1
from phase2a_dag_pipeline.nodes.node2_spatial_layout  import run_node2
from phase2a_dag_pipeline.nodes.node3_colour_attribute import run_node3
from phase2a_dag_pipeline.nodes.node4_typography      import run_node4
from phase2a_dag_pipeline.nodes.node5_brand_quality   import run_node5
from phase2a_dag_pipeline.nodes.node6_coordinator     import run_node6


def _route_after_node1(state: BrandComplianceState) -> str:
    """
    Conditional routing after Node 1.
    Short-circuit to Node 6 if any required object is absent.
    Otherwise proceed to Node 2.
    """
    if state.get("node1_short_circuit", False):
        return "node6"
    return "node2"


def build_graph():
    """Build and compile the MulBrandEval LangGraph DAG."""
    graph = StateGraph(BrandComplianceState)

    # ── Register all nodes ────────────────────────────────────────────────────
    graph.add_node("node0", run_node0)
    graph.add_node("node1", run_node1)
    graph.add_node("node2", run_node2)
    graph.add_node("node3", run_node3)
    graph.add_node("node4", run_node4)
    graph.add_node("node5", run_node5)
    graph.add_node("node6", run_node6)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("node0")

    # ── Fixed edges ───────────────────────────────────────────────────────────
    graph.add_edge("node0", "node1")

    # ── Conditional edge from Node 1 ──────────────────────────────────────────
    graph.add_conditional_edges(
        "node1",
        _route_after_node1,
        {
            "node2": "node2",   # all objects present → full evaluation
            "node6": "node6",   # any object absent  → short-circuit
        },
    )

    # ── Sequential evaluation chain (Nodes 2 → 3 → 4 → 5 → 6) ───────────────
    graph.add_edge("node2", "node3")
    graph.add_edge("node3", "node4")
    graph.add_edge("node4", "node5")
    graph.add_edge("node5", "node6")

    # ── Terminal edge ─────────────────────────────────────────────────────────
    graph.add_edge("node6", END)

    return graph.compile()


# ── Module-level compiled app — import and use directly ───────────────────────
app = build_graph()