"""
Node 5 — Brand Quality (PickScore)

Computes a human-preference-aligned quality score for the generated image
against the prompt text using PickScore (Kirstain et al., 2023).

Score: cosine similarity between PickScore text and image embeddings.
Normalised to [0,1] via (raw_cosine + 1) / 2.
PASS threshold: normalised_score >= 0.20
(lenient floor — this node measures aesthetic quality, not constraint compliance)

No API calls. Model downloaded from HuggingFace on first run (~1.7 GB, cached).

Language note (Point 2 resolution): scores against prompt_text_en, the
English original, regardless of which language's image is being evaluated.
PickScore's text encoder (CLIP-ViT-H, LAION-trained) is predominantly
English-trained — feeding it translated (e.g. Arabic) text would depress
scores for encoder-reliability reasons unrelated to image quality. Locking
to English keeps Node 5 language-invariant, same principle as required_text
in Nodes 1/3/4.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from phase2a_dag_pipeline.state import BrandComplianceState

# ── Model singleton (loaded once per process) ────────────────────────────────
_PROCESSOR = None
_MODEL      = None

PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_NAME     = "yuvalkirstain/PickScore_v1"
PASS_THRESHOLD = 0.20   # normalised score; lenient aesthetic floor


def _load_model():
    global _PROCESSOR, _MODEL
    if _PROCESSOR is None:
        print("  [Node 5] Loading PickScore model (first run — ~1.7 GB download if not cached)...")
        _PROCESSOR = AutoProcessor.from_pretrained(PROCESSOR_NAME)
        _MODEL     = AutoModel.from_pretrained(MODEL_NAME).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = _MODEL.to(device)
        print(f"  [Node 5] Model loaded on {device}.")
    return _PROCESSOR, _MODEL


def _pickscore(prompt: str, image_path: str) -> float:
    """Return normalised PickScore in [0, 1] for one prompt-image pair."""
    processor, model = _load_model()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")

    img_inputs  = processor(images=image,  return_tensors="pt", padding=True).to(device)
    text_inputs = processor(text=prompt,   return_tensors="pt", padding=True,
                            truncation=True, max_length=77).to(device)

    with torch.no_grad():
        img_embs  = model.get_image_features(**img_inputs)
        img_embs  = img_embs  / img_embs.norm(dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        raw_cosine = (text_embs @ img_embs.T).item()   # cosine similarity

    normalised = (raw_cosine + 1.0) / 2.0              # map [-1,1] → [0,1]
    return round(normalised, 4)


def run_node5(state: BrandComplianceState) -> dict:
    """
    Compute PickScore for the generated image.
    Returns state updates for node5_score and node5_pass.
    """
    brief_id   = state.get("brief_id", "UNKNOWN")
    image_path = state["image_path"]
    prompt     = state["prompt_text_en"]   # English-locked (Point 2) — was state["prompt_text"]

    score     = _pickscore(prompt, image_path)
    pass_fail = score >= PASS_THRESHOLD

    print(f"  [{brief_id}] Node 5 {'PASS' if pass_fail else 'FAIL'} — "
          f"PickScore={score:.4f} (threshold={PASS_THRESHOLD})")

    return {
        "node5_score": score,
        "node5_pass":  pass_fail,
    }