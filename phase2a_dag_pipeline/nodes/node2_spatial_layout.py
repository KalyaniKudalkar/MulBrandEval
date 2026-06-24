"""
Node 2 — Spatial Layout (GPT-4o mini directional VQA)

Checks whether required spatial positioning constraints between objects
are satisfied in the generated image.

Uses binary directional VQA: "Is the {obj0} to the {direction} of the {obj1}?"
(Phase 1C calibration: Accuracy=0.7833, target ≥0.50 — PASS)

PASS threshold: mean constraint score >= 0.50
Only executes if Node 1 passed (short-circuit guard from Node 1).
Degenerate constraints (obj0 == obj1) are removed by Node 0 before reaching here.
"""
import base64
import os
from openai import OpenAI
from phase2a_dag_pipeline.state import BrandComplianceState

# ── OpenAI client singleton ───────────────────────────────────────────────────
_CLIENT = None
MODEL   = "gpt-4o-mini"
PASS_THRESHOLD  = 0.50
DIRECTION_WORDS = {"left", "right", "above", "below"}


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _CLIENT


def _encode_image(image_path: str) -> str:
    """Base64-encode image for GPT-4o mini vision input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_constraint(constraint: str) -> tuple[str, str, str] | None:
    """
    Parse a spatial constraint string into (obj0, direction, obj1).

    Handles both formats:
      "smartwatch left of wireless charger"  → left/right (with 'of')
      "wireless earbuds above charging case" → above/below (without 'of')

    Returns None if no direction word found (constraint is skipped).
    """
    parts = constraint.lower().split()
    for i, word in enumerate(parts):
        if word in DIRECTION_WORDS:
            obj0      = " ".join(parts[:i]).strip()
            direction = word
            rest      = parts[i + 1:]
            # strip leading "of" if present (left/right format)
            if rest and rest[0] == "of":
                rest = rest[1:]
            obj1 = " ".join(rest).strip()

            # Degenerate guard (belt-and-suspenders — Node 0 should have caught this)
            assert obj0 != obj1, (
                f"Node 2: Degenerate constraint '{constraint}' — obj0==obj1. "
                f"Node 0 should have filtered this."
            )
            return obj0, direction, obj1

    return None   # no direction word found — skip this constraint


def _ask_spatial(image_b64: str, obj0: str, direction: str, obj1: str) -> float:
    """
    Ask GPT-4o mini: 'Is the {obj0} to the {direction} of the {obj1}?'
    Returns 1.0 for yes, 0.0 for no.
    Same directional binary VQA format as Phase 1C Node 2.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/png;base64,{image_b64}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Is the {obj0} to the {direction} of the {obj1} "
                            f"in this image? Answer only 'yes' or 'no'."
                        ),
                    },
                ],
            }
        ],
        max_tokens=5,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip().lower()
    return 1.0 if answer.startswith("yes") else 0.0


def run_node2(state: BrandComplianceState) -> dict:
    """
    Check all spatial constraints for the generated image.
    Returns state updates for node2_* fields.
    """
    brief_id            = state.get("brief_id", "UNKNOWN")
    image_path          = state["image_path"]
    spatial_constraints = state["spatial_constraints"]

    # No constraints defined — pass trivially
    if not spatial_constraints:
        print(f"  [{brief_id}] Node 2 PASS — no spatial constraints defined")
        return {
            "node2_per_constraint": {},
            "node2_score":          1.0,
            "node2_pass":           True,
        }

    image_b64      = _encode_image(image_path)
    per_constraint = {}
    skipped        = 0

    for constraint in spatial_constraints:
        parsed = _parse_constraint(constraint)
        if parsed is None:
            print(f"  [{brief_id}] Node 2: Skipping unparseable constraint '{constraint}'")
            skipped += 1
            continue
        obj0, direction, obj1 = parsed
        score = _ask_spatial(image_b64, obj0, direction, obj1)
        per_constraint[constraint] = score

    if not per_constraint:
        # All constraints were unparseable — treat as no constraints
        mean_score = 1.0
        pass_fail  = True
    else:
        mean_score = round(
            sum(per_constraint.values()) / len(per_constraint), 4
        )
        pass_fail = mean_score >= PASS_THRESHOLD

    print(f"  [{brief_id}] Node 2 {'PASS' if pass_fail else 'FAIL'} — "
          f"constraints={per_constraint}, mean={mean_score:.4f} "
          f"(threshold={PASS_THRESHOLD}, skipped={skipped})")

    return {
        "node2_per_constraint": per_constraint,
        "node2_score":          mean_score,
        "node2_pass":           pass_fail,
    }