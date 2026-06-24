"""
BrandComplianceState — shared state TypedDict for the MulBrandEval LangGraph DAG.
All nodes read from and write to this state object.
"""
from typing import TypedDict, Optional


class BrandComplianceState(TypedDict):

    # ── Inputs (set before graph runs) ──────────────────────────────────────
    brief_id:    str
    industry:    str
    brand_name:  str
    prompt_text: str
    image_path:  str
    model:       str    # "sd15" | "flux"
    language:    str    # "en" | "de" | "fr" | "es" | "ar"

    # ── Node 0 — Prompt Parser ───────────────────────────────────────────────
    # Initially strings from CSV; Node 0 parses into Python objects
    required_objects:    list   # ["smartwatch", "wireless charger"]
    colour_attributes:   dict   # {"smartwatch": "silver", ...}
    spatial_constraints: list   # ["smartwatch left of wireless charger"]
    required_text:       list   # ["NOVA"]
    style_descriptors:   list   # ["product photography", "minimalist"]

    # ── Node 1 — Object Presence ─────────────────────────────────────────────
    node1_per_object:    Optional[dict]   # {"smartwatch": True, "charger": False}
    node1_score:         Optional[float]  # fraction present: 1/2 = 0.5
    node1_pass:          Optional[bool]   # True only if ALL objects present
    node1_short_circuit: Optional[bool]   # True → skip Nodes 2 and 3

    # ── Node 2 — Spatial Layout ──────────────────────────────────────────────
    node2_per_constraint: Optional[dict]  # {"smartwatch left of charger": True}
    node2_score:          Optional[float]
    node2_pass:           Optional[bool]

    # ── Node 3 — Colour Attribute ────────────────────────────────────────────
    node3_per_object: Optional[dict]   # {"smartwatch": True, "charger": False}
    node3_score:      Optional[float]
    node3_pass:       Optional[bool]

    # ── Node 4 — Typography ──────────────────────────────────────────────────
    node4_extracted:  Optional[str]    # best OCR match found in image
    node4_similarity: Optional[float]  # Levenshtein similarity 0-1
    node4_pass:       Optional[bool]
    node4_skipped:    Optional[bool]   # True for SD v1.5 (architectural limitation)

    # ── Node 5 — Brand Quality ───────────────────────────────────────────────
    node5_score: Optional[float]   # PickScore 0-1
    node5_pass:  Optional[bool]

    # ── Node 6 — Coordinator / Aggregator ────────────────────────────────────
    compliance_score:  Optional[float]  # weighted sum 0-1
    compliance_report: Optional[dict]   # full per-node JSON report
    failure_diagnosis: Optional[str]    # human-readable failure summary