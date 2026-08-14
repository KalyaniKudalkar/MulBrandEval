"""
Node 6 (Flat variant) — Coordinator / Aggregator for Configuration C
=======================================================================
Used ONLY by pipeline_flat.py (Configuration C — flat pipeline, no
short-circuit). Do NOT use this in the main DAG (pipeline.py) — that
one correctly uses node6_coordinator.py.

WHY THIS FILE EXISTS (do not merge with node6_coordinator.py):
node6_coordinator.py hardcodes node2_score / node3_score to 0.0 whenever
node1_short_circuit is True — this is correct for Configuration B, where
Nodes 2/3 never execute in that case (LangGraph routes straight to Node 6).
But in Configuration C, Nodes 2-5 execute unconditionally for EVERY image,
including ones where Node 1 failed. If this flat pipeline reused the
original coordinator, it would silently zero out real, computed Node 2/3
scores just because node1_short_circuit=True — making Configuration C's
compliance scores collapse back toward Configuration B's, defeating the
entire point of the ablation (testing what a flat/no-dependency evaluator
"sees" that the DAG's short-circuit logic prevents it from computing).

This coordinator trusts whatever each node actually returned. The only
skip logic preserved is Node 4's SD v1.5 architectural skip (model-based,
not short-circuit-based — Node 4 always skips for sd15 regardless of
pipeline structure, per Phase 1C accuracy=0.1667 finding).

Node weights are IDENTICAL to node6_coordinator.py (BASE_WEIGHTS) —
weights must match Configuration B exactly for the compliance scores to
be comparable in the A/B/C ablation.
"""
from phase2a_dag_pipeline.state import BrandComplianceState

BASE_WEIGHTS = {
    "node1": 0.30,
    "node2": 0.20,
    "node3": 0.20,
    "node4": 0.20,
    "node5": 0.10,
}


def _get_weights() -> dict:
    return BASE_WEIGHTS.copy()


def _build_diagnosis_flat(
    node1_pass: bool,
    node1_short_circuit: bool,
    node2_pass, node3_pass, node4_pass, node4_arch_skipped,
    node5_pass, brief_id: str, model: str,
) -> str:
    """
    Human-readable diagnosis for the flat pipeline. Unlike the DAG version,
    this reports Node 2/3/4/5 outcomes even when node1_short_circuit=True,
    since those nodes genuinely ran and produced real results here.
    """
    failures = []

    if not node1_pass:
        failures.append("Node1: required objects absent")
        if node1_short_circuit:
            failures.append(
                "(informational only — flat pipeline: Nodes 2-5 still executed)"
            )

    if node2_pass is False:
        failures.append("Node2: spatial constraints not satisfied")
    if node3_pass is False:
        failures.append("Node3: colour attributes not satisfied")

    if node4_arch_skipped:
        failures.append(
            "Node4: SD v1.5 architectural limitation — text rendering "
            "unreliable (Phase 1C accuracy=0.1667); scored as 0.0"
        )
    elif node4_pass is False:
        failures.append("Node4: required text not legibly rendered")

    if node5_pass is False:
        failures.append("Node5: brand quality below aesthetic threshold")

    if not failures:
        return f"[{brief_id}] COMPLIANT — all constraints satisfied"
    return f"[{brief_id}] NON-COMPLIANT — " + "; ".join(failures)


def run_node6_flat(state: BrandComplianceState) -> dict:
    """
    Aggregate all node scores for the FLAT pipeline (Configuration C).
    No short-circuit zeroing — every node's actual output is used as-is.
    """
    brief_id = state.get("brief_id", "UNKNOWN")
    model    = state.get("model", "")

    node1_pass          = state.get("node1_pass", False)
    node1_score         = state.get("node1_score", 0.0)
    node1_short_circuit = state.get("node1_short_circuit", False)  # informational only here

    # Nodes 2 and 3 ALWAYS ran in the flat pipeline — use their real output directly.
    node2_pass  = state.get("node2_pass")
    node2_score = state.get("node2_score", 0.0)

    node3_pass  = state.get("node3_pass")
    node3_score = state.get("node3_score", 0.0)

    # Node 4: only skip reason preserved is the SD v1.5 architectural one.
    node4_arch_skipped = state.get("node4_skipped", False)  # True only for SD v1.5
    node4_pass  = state.get("node4_pass")
    node4_score = state.get("node4_similarity", 0.0) if not node4_arch_skipped else 0.0

    # Node 5 always ran in the flat pipeline.
    node5_pass  = state.get("node5_pass")
    node5_score = state.get("node5_score", 0.0)

    weights = _get_weights()

    compliance_score = round(
        node1_score * weights["node1"] +
        node2_score * weights["node2"] +
        node3_score * weights["node3"] +
        node4_score * weights["node4"] +
        node5_score * weights["node5"],
        4,
    )

    compliance_report = {
        "brief_id": brief_id,
        "model":    model,
        "language": state.get("language", "en"),
        "node1": {
            "pass":          node1_pass,
            "score":         node1_score,
            "per_object":    state.get("node1_per_object"),
            "short_circuit": node1_short_circuit,  # informational — did NOT gate anything here
            "weight":        weights["node1"],
        },
        "node2": {
            "pass":           node2_pass,
            "score":          node2_score,
            "per_constraint": state.get("node2_per_constraint"),
            "skipped":        False,   # never skipped in the flat pipeline
            "weight":         weights["node2"],
        },
        "node3": {
            "pass":       node3_pass,
            "score":      node3_score,
            "per_object": state.get("node3_per_object"),
            "skipped":    False,       # never skipped in the flat pipeline
            "weight":     weights["node3"],
        },
        "node4": {
            "pass":           node4_pass,
            "score":          node4_score,
            "extracted":      state.get("node4_extracted"),
            "similarity":     state.get("node4_similarity"),
            "skipped":        node4_arch_skipped,
            "skipped_reason": "sd15_architectural" if node4_arch_skipped else None,
            "weight":         weights["node4"],
            "clcg_interpretable": not node4_arch_skipped,
        },
        "node5": {
            "pass":   node5_pass,
            "score":  node5_score,
            "weight": weights["node5"],
        },
        "compliance_score": compliance_score,
        "weights_used":     weights,
    }

    failure_diagnosis = _build_diagnosis_flat(
        node1_pass, node1_short_circuit,
        node2_pass, node3_pass, node4_pass, node4_arch_skipped,
        node5_pass, brief_id, model,
    )

    print(f"  [{brief_id}] Node 6 (flat) — compliance_score={compliance_score:.4f} "
          f"(model={model})")
    print(f"  {failure_diagnosis}")

    return {
        "compliance_score":  compliance_score,
        "compliance_report": compliance_report,
        "failure_diagnosis": failure_diagnosis,
    }