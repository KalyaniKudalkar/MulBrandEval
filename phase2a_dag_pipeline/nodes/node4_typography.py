"""
Node 4 — Typography (EasyOCR + Levenshtein)

Checks whether the required brand text is legibly rendered in the generated image.
Uses EasyOCR to extract all text regions, then picks the best Levenshtein
similarity match against the required text string.

PASS threshold: best_similarity >= 0.50 (from Phase 1C calibration)

SD v1.5 routing: architecture cannot reliably render text (Phase 1C finding:
accuracy=0.1667). Node is skipped for SD v1.5 and flagged as architectural
limitation rather than scored — avoids penalising the pipeline for a known
model constraint.
"""
import easyocr
from Levenshtein import ratio as lev_ratio
from phase2a_dag_pipeline.state import BrandComplianceState

# ── EasyOCR reader singleton (loaded once per process) ───────────────────────
_READER = None
PASS_THRESHOLD = 0.50   # from Phase 1C calibration


def _get_reader():
    global _READER
    if _READER is None:
        print("  [Node 4] Initialising EasyOCR reader (en)...")
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("  [Node 4] EasyOCR ready.")
    return _READER


def _best_match(extracted_texts: list, required: str) -> tuple[str, float]:
    """
    Return (best_text, best_similarity) from all OCR-detected regions
    against the required string. Case-insensitive comparison.
    """
    if not extracted_texts:
        return "", 0.0

    required_lower = required.lower()
    best_text  = ""
    best_score = 0.0

    for text in extracted_texts:
        score = lev_ratio(text.lower(), required_lower)
        if score > best_score:
            best_score = score
            best_text  = text

    return best_text, round(best_score, 4)


def run_node4(state: BrandComplianceState) -> dict:
    """
    Run typography check on the generated image.
    Returns state updates for node4_* fields.
    """
    brief_id      = state.get("brief_id", "UNKNOWN")
    model         = state.get("model", "")
    image_path    = state["image_path"]
    required_text = state["required_text"]

    # ── SD v1.5 short-circuit (architectural limitation) ─────────────────────
    if model == "sd15":
        print(f"  [{brief_id}] Node 4 SKIP — SD v1.5 architectural limitation "
              f"(text rendering unreliable, Phase 1C accuracy=0.1667)")
        return {
            "node4_extracted":  "N/A — SD v1.5 architectural limitation",
            "node4_similarity": None,
            "node4_pass":       None,
            "node4_skipped":    True,
        }

    # ── Extract required text (first brand name in list) ─────────────────────
    target = required_text[0] if isinstance(required_text, list) else required_text
    target = str(target).strip()

    # ── Run EasyOCR ──────────────────────────────────────────────────────────
    reader = _get_reader()
    try:
        raw_results = reader.readtext(image_path, detail=0)  # text strings only
    except Exception as e:
        print(f"  [{brief_id}] Node 4 ERROR — OCR failed: {e}")
        return {
            "node4_extracted":  "",
            "node4_similarity": 0.0,
            "node4_pass":       False,
            "node4_skipped":    False,
        }

    # ── Find best match ───────────────────────────────────────────────────────
    best_text, best_sim = _best_match(raw_results, target)
    pass_fail = best_sim >= PASS_THRESHOLD

    print(f"  [{brief_id}] Node 4 {'PASS' if pass_fail else 'FAIL'} — "
          f"required='{target}', extracted='{best_text}', "
          f"similarity={best_sim:.4f} (threshold={PASS_THRESHOLD})")

    return {
        "node4_extracted":  best_text,
        "node4_similarity": best_sim,
        "node4_pass":       pass_fail,
        "node4_skipped":    False,
    }