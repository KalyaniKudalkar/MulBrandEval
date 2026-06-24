"""
Node 6 — Coordinator / Aggregator

Collects all node outputs and produces:
  1. Weighted compliance score (0-1)
  2. Per-node compliance report (structured dict)
  3. Human-readable failure diagnosis

═══════════════════════════════════════════════════════════════
NODE 4 — TYPOGRAPHY: DUAL-OUTPUT TREATMENT
(Decided: June 2026 — documented for thesis Section 4.4 / 5.x)

Two separate outputs are affected differently by Node 4:

OUTPUT 1 — Compliance Score (Phase 2A):
  SD v1.5 receives 0.0 on Node 4 (same weights for both models).
  SD v1.5 architecturally cannot render text (Phase 1C: accuracy=0.1667).
  This is a genuine compliance failure — the brief required text that
  was not rendered. Identical weights preserve cross-model comparability.

OUTPUT 2 — Per-Node CLCG Heatmap (Phase 2B):
  SD v1.5 Node 4 CLCG cells are marked "N/A — Architecturally undefined".
  CLCG measures language-induced degradation. If the English baseline is
  already ~0.17 due to an architectural limitation, the CLCG value is not
  a linguistic finding — it is noise. Reporting it as a number would be
  misleading. FLUX Node 4 CLCG is fully interpretable and reported normally.
  This is handled in the Phase 2B analysis script, not here — the pipeline
  correctly records node4_skipped=True for SD v1.5 for downstream use.

STANDALONE ARCHITECTURAL FINDING (reported separately in thesis):
  Node 4 reveals a binary architectural capability gap:
  FLUX reliable text rendering EN=0.9167 vs SD v1.5 EN=0.1667.
  Node 4 CLCG is only interpretable for FLUX, where Arabic prompts
  show the largest degradation due to RTL script rendering demands.
═══════════════════════════════════════════════════════════════

Node weights (sum = 1.0) — IDENTICAL for both models:
  Node 1 — Object Presence:   0.30
  Node 2 — Spatial Layout:    0.20
  Node 3 — Colour Attribute:  0.20
  Node 4 — Typography:        0.20  (0.0 for SD v1.5 — compliance failure)
  Node 5 — Brand Quality:     0.10
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
    """
    Return BASE_WEIGHTS unchanged for both models.
    Node 4 weight is always 0.20 — SD v1.5 simply scores 0.0 on it.
    No redistribution. Both models evaluated on identical criteria.
    """
    return BASE_WEIGHTS.copy()


def _build_diagnosis(
    node1_pass: bool,
    node1_short_circuit: bool,
    node2_pass,
    node3_pass,
    node4_pass,
    node4_arch_skipped: bool,
    node4_sc_skipped: bool,
    node5_pass,
    brief_id: str,
    model: str,
) -> str:
    """Build a human-readable failure diagnosis string."""
    failures = []

    if not node1_pass:
        failures.append(
            "Node1: required objects absent (short-circuit triggered)"
        )
    if node1_short_circuit:
        failures.append(
            "Node2/Node3: skipped — object presence failed upstream"
        )
    elif node2_pass is False:
        failures.append("Node2: spatial constraints not satisfied")

    if node3_pass is False and not node1_short_circuit:
        failures.append("Node3: colour attributes not satisfied")

    if node4_arch_skipped:
        failures.append(
            "Node4: SD v1.5 architectural limitation — text rendering "
            "unreliable (Phase 1C accuracy=0.1667); scored as 0.0 in "
            "compliance score; excluded from CLCG heatmap in Phase 2B"
        )
    elif node4_sc_skipped:
        pass  # Not executed — already covered by short-circuit message above
    elif node4_pass is False:
        failures.append("Node4: required text not legibly rendered")

    # Only report Node 5 if it actually ran (None = not executed due to short-circuit)
    if node5_pass is not None and not node5_pass:
        failures.append("Node5: brand quality below aesthetic threshold")

    if not failures:
        return f"[{brief_id}] COMPLIANT — all constraints satisfied"
    return f"[{brief_id}] NON-COMPLIANT — " + "; ".join(failures)


def run_node6(state: BrandComplianceState) -> dict:
    """
    Aggregate all node scores into a final compliance report.
    Returns state updates for compliance_score, compliance_report,
    and failure_diagnosis.
    """
    brief_id = state.get("brief_id", "UNKNOWN")
    model    = state.get("model", "")

    # ── Retrieve node outputs ─────────────────────────────────────────────────
    node1_pass          = state.get("node1_pass",          False)
    node1_score         = state.get("node1_score",         0.0)
    node1_short_circuit = state.get("node1_short_circuit", False)

    # Nodes 2 and 3: 0.0 if short-circuited (objects absent → cannot comply)
    node2_pass  = state.get("node2_pass")
    node2_score = state.get("node2_score", 0.0) if not node1_short_circuit else 0.0

    node3_pass  = state.get("node3_pass")
    node3_score = state.get("node3_score", 0.0) if not node1_short_circuit else 0.0

    # Distinguish skip reason: SD v1.5 architectural vs Node 1 short-circuit
    node4_arch_skipped        = state.get("node4_skipped", False)  # True only for SD v1.5
    node4_sc_skipped          = node1_short_circuit                 # True if short-circuited
    node4_effectively_skipped = node4_arch_skipped or node4_sc_skipped
    node4_pass  = state.get("node4_pass")
    node4_score = state.get("node4_similarity", 0.0) if not node4_effectively_skipped else 0.0

    # Node 5: None means never executed (short-circuited), False means ran and failed
    node5_pass  = state.get("node5_pass")   # intentionally no default — None = not run
    node5_score = state.get("node5_score", 0.0) if node5_pass is not None else 0.0

    # ── Weighted compliance score (identical weights for both models) ──────────
    weights = _get_weights()

    compliance_score = round(
        node1_score * weights["node1"] +
        node2_score * weights["node2"] +
        node3_score * weights["node3"] +
        node4_score * weights["node4"] +
        node5_score * weights["node5"],
        4,
    )

    # ── Per-node report ───────────────────────────────────────────────────────
    compliance_report = {
        "brief_id":  brief_id,
        "model":     model,
        "language":  state.get("language", "en"),
        "node1": {
            "pass":          node1_pass,
            "score":         node1_score,
            "per_object":    state.get("node1_per_object"),
            "short_circuit": node1_short_circuit,
            "weight":        weights["node1"],
        },
        "node2": {
            "pass":           node2_pass,
            "score":          node2_score,
            "per_constraint": state.get("node2_per_constraint"),
            "skipped":        node1_short_circuit,
            "weight":         weights["node2"],
        },
        "node3": {
            "pass":       node3_pass,
            "score":      node3_score,
            "per_object": state.get("node3_per_object"),
            "skipped":    node1_short_circuit,
            "weight":     weights["node3"],
        },
        "node4": {
            "pass":       node4_pass,
            "score":      node4_score,
            "extracted":  state.get("node4_extracted"),
            "similarity": state.get("node4_similarity"),
            "skipped":    node4_effectively_skipped,
            "skipped_reason": (
                "sd15_architectural" if node4_arch_skipped
                else "short_circuit" if node4_sc_skipped
                else None
            ),
            "weight":     weights["node4"],
            # False only for SD v1.5 — short-circuit skip does not affect CLCG interpretability
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

    # ── Failure diagnosis ─────────────────────────────────────────────────────
    failure_diagnosis = _build_diagnosis(
        node1_pass, node1_short_circuit,
        node2_pass, node3_pass,
        node4_pass, node4_arch_skipped, node4_sc_skipped,
        node5_pass, brief_id, model,
    )

    print(f"  [{brief_id}] Node 6 — compliance_score={compliance_score:.4f} "
          f"(model={model})")
    print(f"  {failure_diagnosis}")

    return {
        "compliance_score":  compliance_score,
        "compliance_report": compliance_report,
        "failure_diagnosis": failure_diagnosis,
    }