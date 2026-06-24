"""
Node 3 — Colour Attribute (GPT-4o mini confirmatory VQA)

Checks whether each required object appears in its specified colour.
Uses confirmatory binary VQA: "Is the {object} {colour} in this image?"
(Node 3r recalibrated pattern from Phase 1C: ρ=0.5063, p<0.0001)

PASS threshold: mean colour score >= 0.60
Only executes if Node 1 passed (all objects present).
If node1_short_circuit=True this node is skipped by the DAG — no need
to check colours of objects that aren't in the image.
"""
import base64
import os
from openai import OpenAI
from phase2a_dag_pipeline.state import BrandComplianceState

# ── OpenAI client singleton ───────────────────────────────────────────────────
_CLIENT = None
MODEL   = "gpt-4o-mini"
PASS_THRESHOLD = 0.60


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _CLIENT


def _encode_image(image_path: str) -> str:
    """Base64-encode image for GPT-4o mini vision input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _ask_colour_present(image_b64: str, object_name: str, colour: str) -> float:
    """
    Ask GPT-4o mini: 'Is the {object} {colour} in this image?'
    Returns 1.0 for yes, 0.0 for no.
    Confirmatory format from Phase 1C Node 3r calibration.
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
                            f"Is the {object_name} {colour} in this image? "
                            f"Answer only 'yes' or 'no'."
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


def run_node3(state: BrandComplianceState) -> dict:
    """
    Check colour compliance for each required object.
    Returns state updates for node3_* fields.
    """
    brief_id          = state.get("brief_id", "UNKNOWN")
    image_path        = state["image_path"]
    colour_attributes = state["colour_attributes"]  # {"smartwatch": "silver", ...}

    image_b64  = _encode_image(image_path)
    per_object = {}

    for obj, colour in colour_attributes.items():
        score = _ask_colour_present(image_b64, obj, colour)
        per_object[obj] = score

    mean_score = round(
        sum(per_object.values()) / len(per_object), 4
    ) if per_object else 0.0

    pass_fail = mean_score >= PASS_THRESHOLD

    print(f"  [{brief_id}] Node 3 {'PASS' if pass_fail else 'FAIL'} — "
          f"{per_object}, mean={mean_score:.4f} (threshold={PASS_THRESHOLD})")

    return {
        "node3_per_object": per_object,
        "node3_score":      mean_score,
        "node3_pass":       pass_fail,
    }