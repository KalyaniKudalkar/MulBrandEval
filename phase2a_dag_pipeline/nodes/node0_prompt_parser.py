"""
Node 0 — Prompt Parser

Phase 2A: Parses pre-structured JSON string columns from the MBB CSV row
into Python objects and validates them before any downstream node runs.

Phase 2B note: GPT-4o mini extraction path will be activated here for
non-English natural language briefs where columns arrive as raw text.
No API calls are made in Phase 2A.
"""
import json
from phase2a_dag_pipeline.state import BrandComplianceState


def _parse_json_field(value, field_name: str):
    """Parse a JSON string column into a Python object."""
    if isinstance(value, (list, dict)):
        return value  # already parsed (e.g. in unit tests)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(
            f"Node 0: Failed to parse field '{field_name}': {e}\nValue: {value!r}"
        )


def _clean_spatial_constraints(constraints: list, brief_id: str) -> list:
    """
    Remove degenerate constraints where obj0 == obj1.
    Guard introduced from Phase 1C Node 2 finding (bench/bench, bear/bear cases).
    """
    direction_words = {"left", "right", "above", "below"}
    clean = []
    for c in constraints:
        parts = c.lower().split()
        dirs_found = [w for w in parts if w in direction_words]
        if dirs_found:
            idx = parts.index(dirs_found[0])
            obj0 = " ".join(parts[:idx]).strip()
            # skip direction word + "of"
            rest = parts[idx + 1:]
            if rest and rest[0] == "of":
                rest = rest[1:]
            obj1 = " ".join(rest).strip()
            if obj0 == obj1:
                print(f"  [{brief_id}] Node 0: Skipping degenerate constraint "
                      f"'{c}' (obj0 == obj1)")
                continue
        clean.append(c)
    return clean


def run_node0(state: BrandComplianceState) -> dict:
    """
    Parse and validate all structured fields from the brand brief.
    Returns a dict of state updates for LangGraph.
    """
    brief_id = state.get("brief_id", "UNKNOWN")

    # ── Parse JSON string fields ─────────────────────────────────────────────
    required_objects    = _parse_json_field(state["required_objects"],    "required_objects")
    colour_attributes   = _parse_json_field(state["colour_attributes"],   "colour_attributes")
    spatial_constraints = _parse_json_field(state["spatial_constraints"], "spatial_constraints")
    required_text       = _parse_json_field(state["required_text"],       "required_text")
    style_descriptors   = _parse_json_field(state["style_descriptors"],   "style_descriptors")

    # ── Validate ──────────────────────────────────────────────────────────────
    assert isinstance(required_objects, list)  and len(required_objects) >= 1, \
        f"[{brief_id}] required_objects must be a non-empty list"
    assert isinstance(colour_attributes, dict) and len(colour_attributes) >= 1, \
        f"[{brief_id}] colour_attributes must be a non-empty dict"
    assert isinstance(spatial_constraints, list), \
        f"[{brief_id}] spatial_constraints must be a list"
    assert isinstance(required_text, list)     and len(required_text) >= 1, \
        f"[{brief_id}] required_text must be a non-empty list"
    assert isinstance(style_descriptors, list), \
        f"[{brief_id}] style_descriptors must be a list"

    # ── Clean spatial constraints ─────────────────────────────────────────────
    spatial_constraints = _clean_spatial_constraints(spatial_constraints, brief_id)

    print(f"  [{brief_id}] Node 0 PASS — "
          f"objects={required_objects}, "
          f"colours={list(colour_attributes.keys())}, "
          f"constraints={spatial_constraints}, "
          f"text={required_text}")

    return {
        "required_objects":    required_objects,
        "colour_attributes":   colour_attributes,
        "spatial_constraints": spatial_constraints,
        "required_text":       required_text,
        "style_descriptors":   style_descriptors,
    }