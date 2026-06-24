"""
Node 1 — Object Presence (GPT-4o mini VQA)

Checks whether every required object from the brand brief is clearly
visible in the generated image. Uses binary yes/no VQA via GPT-4o mini
vision at detail=low (Phase 1C calibration: F1=0.8352, P=1.0, R=0.7171).

PASS: ALL required objects present (score = 1.0)
FAIL: ANY object absent → node1_short_circuit=True → Nodes 2 and 3 skipped
(spatial and colour checks are meaningless if the object isn't in the image)

Short-circuit rationale: prevents cascading false failures downstream.
"""
import base64
import os
from openai import OpenAI
from phase2a_dag_pipeline.state import BrandComplianceState

# ── OpenAI client singleton ───────────────────────────────────────────────────
_CLIENT = None
MODEL   = "gpt-4o-mini"


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _CLIENT


def _encode_image(image_path: str) -> str:
    """Base64-encode image for GPT-4o mini vision input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _ask_object_present(image_b64: str, object_name: str) -> bool:
    """
    Ask GPT-4o mini: 'Is [object] clearly visible in this image?'
    Returns True if yes, False if no.
    Same prompt format as Phase 1C Node 1 (detail=low).
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
                            f"Is a {object_name} clearly visible in this image? "
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
    return answer.startswith("yes")


def run_node1(state: BrandComplianceState) -> dict:
    """
    Check presence of every required object in the generated image.
    Returns state updates for node1_* fields.
    """
    brief_id         = state.get("brief_id", "UNKNOWN")
    image_path       = state["image_path"]
    required_objects = state["required_objects"]

    image_b64    = _encode_image(image_path)
    per_object   = {}
    all_present  = True

    for obj in required_objects:
        present = _ask_object_present(image_b64, obj)
        per_object[obj] = present
        if not present:
            all_present = False

    # Score = fraction of objects present
    n_present = sum(per_object.values())
    score     = round(n_present / len(required_objects), 4) if required_objects else 0.0

    # Short-circuit if any object missing
    short_circuit = not all_present

    print(f"  [{brief_id}] Node 1 {'PASS' if all_present else 'FAIL (SHORT-CIRCUIT)'} — "
          f"{per_object}, score={score:.4f}")

    return {
        "node1_per_object":    per_object,
        "node1_score":         score,
        "node1_pass":          all_present,
        "node1_short_circuit": short_circuit,
    }