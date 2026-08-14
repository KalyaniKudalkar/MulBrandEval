"""
MulBrandEval — Flat LangGraph DAG Pipeline
Configuration C — Flat pipeline, no dependency ordering

DAG execution order (UNCONDITIONAL — every node runs on every image):
  Node 0 (Prompt Parser)
    → Node 1 (Object Presence)
        → Node 2 (Spatial Layout)      [runs even if Node 1 failed]
            → Node 3 (Colour Attribute) [runs even if Node 1 failed]
                → Node 4 (Typography)
                    → Node 5 (Brand Quality)
                        → Node 6 (Coordinator — flat variant)
                            → END

This is the direct comparator for Configuration B (pipeline.py): the ONLY
structural difference between this file and pipeline.py is the removal of
the conditional edge after Node 1. Node 1 still computes its score and
short_circuit flag (informational), but nothing downstream reads it to
skip execution. This isolates dependency-ordering as the single variable
under test for RQ3 ("does the DAG's short-circuit logic outperform a flat
evaluation that answers every question regardless of upstream failures?").

Uses node6_coordinator_flat.run_node6_flat, NOT node6_coordinator.run_node6
— see that file's docstring for why the original coordinator cannot be
reused here (it hardcodes zeroing of Node 2/3 scores based on
node1_short_circuit, which would silently corrupt this ablation).

All Node 0-5 functions are imported unchanged from the main DAG package —
Configuration C uses the exact same underlying node logic/calibration as
Configuration B, varying ONLY the routing structure.
"""
from langgraph.graph import StateGraph, END
from phase2a_dag_pipeline.state import BrandComplianceState
from phase2a_dag_pipeline.nodes.node0_prompt_parser  import run_node0
from phase2a_dag_pipeline.nodes.node1_object_presence import run_node1
from phase2a_dag_pipeline.nodes.node2_spatial_layout  import run_node2
from phase2a_dag_pipeline.nodes.node3_colour_attribute import run_node3
from phase2a_dag_pipeline.nodes.node4_typography      import run_node4
from phase2a_dag_pipeline.nodes.node5_brand_quality   import run_node5
from phase2a_dag_pipeline.nodes.node6_coordinator_flat import run_node6_flat


def build_flat_graph():
    """Build and compile the flat (Configuration C) MulBrandEval pipeline."""
    graph = StateGraph(BrandComplianceState)

    graph.add_node("node0", run_node0)
    graph.add_node("node1", run_node1)
    graph.add_node("node2", run_node2)
    graph.add_node("node3", run_node3)
    graph.add_node("node4", run_node4)
    graph.add_node("node5", run_node5)
    graph.add_node("node6", run_node6_flat)

    graph.set_entry_point("node0")

    # ── Fixed, unconditional edges — no routing decision after Node 1 ────────
    graph.add_edge("node0", "node1")
    graph.add_edge("node1", "node2")   # always proceeds, regardless of node1 result
    graph.add_edge("node2", "node3")
    graph.add_edge("node3", "node4")
    graph.add_edge("node4", "node5")
    graph.add_edge("node5", "node6")
    graph.add_edge("node6", END)

    return graph.compile()


# ── Module-level compiled app — import and use directly ───────────────────────
app = build_flat_graph()