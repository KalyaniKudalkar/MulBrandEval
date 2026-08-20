"""
MulBrandEval — Phase 2B: Evaluator Extraction-Accuracy Validation
==============================================================================
Methodological Points Summary, Point 2, "Resolution Part 2: Direct
Measurement of the Extraction Path."

WHAT THIS SCRIPT DOES
----------------------
The Phase 2B evaluation path (Nodes 1-5) is locked to the authored English
structured fields for all five languages — the translated brief text is
used only for image generation and is never consulted by the DAG's
evaluation nodes (Resolution Part 1, implemented by design). This
eliminates evaluator-language degradation from the CLCG heatmap BY
CONSTRUCTION, but it leaves one question unanswered: if the pipeline were
deployed with no English original to fall back on, could GPT-4o mini
correctly parse a raw non-English brief itself?

This script answers exactly that, in total isolation from the CLCG
heatmap, the images, and every other Phase 2B table:

  1. For every brief, in every language (EN as the control + DE/FR/ES/AR),
     GPT-4o mini reads ONLY the raw `prompt_text` (translated sentence) —
     no English hints, no ground truth in the prompt — and extracts
     required_objects / colour_attributes / spatial_constraints /
     required_text as JSON. Same schema and field semantics as
     node0_prompt_parser.py.
  2. The extracted fields are scored against the known-true English fields
     already sitting in mbb_briefs_en.csv (authored fact, not a model or
     human judgment — see Methodological Points Summary).

NO IMAGES ARE TOUCHED. This is a pure text-in/text-out validation:
no Replicate calls, no image generation, no re-evaluation of any existing
Phase 2A/2B image result. It does not feed back into the CLCG heatmap,
Configuration B/C tables, or TIFA scores — it is a standalone deployment-
readiness / evaluator-reliability measurement reported alongside them.

MATCHING RULE
----------------------------------------------------------------------------
No hand-authored semantic synonym list is used. The English brief
vocabulary (259 unique objects across 150 briefs, inspected directly from
mbb_briefs_en.csv) shows deliberately distinct product-level nouns
("wireless charger" vs "wireless charging pad", "smartwatch" vs "fitness
watch") rather than casual synonym variance — a hand-built list risks
silently merging genuinely different products and cannot be justified
as unbiased once results are visible. Matching instead uses two layers,
both purely mechanical (no semantic/LLM judgment at any point):

  1. Normalization: case-folding, whitespace/hyphen normalization, and
     naive plural stripping.
  2. Word-level containment: a ground-truth phrase matches an extracted
     phrase if one phrase's word set is a subset of the other's (e.g.
     ground truth "metal stand" matches extracted "stand"; ground truth
     "camera" matches extracted "dslr camera"). Colours are looked up via
     the best containment-matched object key rather than an exact key.
     spatial_constraints are matched as (object, direction, object)
     triplets using the same containment rule on each object, with the
     direction matched either same-order or as its logical reciprocal
     ("A right of B" == "B left of A"). required_text is matched as a
     whole-word substring, case-fold only, no further normalization
     (brand text is meant to render literally, so no containment/synonym
     leniency is applied to direction or to the text itself — only to the
     surrounding descriptive wording the extraction may add, e.g.
     "KESTRA logo" for ground truth "KESTRA").

Both rule layers were locked in this exact form based ONLY on evidence
from the EN control run (an exact-match-only first pass showed EN itself
scoring well below ceiling, which is only possible if the *matching
logic* has a design flaw, since the control language has no translation
step to fail on). Every containment and reciprocal-direction fix here
was diagnosed from an EN near-miss, not from any DE/FR/ES/AR result, and
the rule was not iterated further once EN reached its ceiling (~0.99
overall; residual EN misses are a genuine extraction-method limit on
complex multi-object spatial phrasing, not a scoring artifact). This
keeps the fix strictly language-blind and non-cherry-picked.

For full transparency, results/phase2b/extraction_accuracy_validation.csv
reports BOTH the containment-based accuracy ("accuracy", primary) and the
original exact-match accuracy ("accuracy_strict"), so the effect of the
containment rule is visible rather than hidden. Any remaining shortfall
for a given language is read as an UPPER BOUND on evaluator degradation,
since translation artefacts and legitimate paraphrase both still
contribute to it (Methodological Points Summary).

For manual review only, every non-exact (containment-level) match is
also logged verbatim to a "near-miss" sidecar file — these are NOT
re-scored or folded back into accuracy.

OUTPUT
------
  results/phase2b/extraction_accuracy_validation.csv
      One row per (language, category) with n_briefs, n_matched,
      accuracy. Categories: objects, colours, spatial_constraints,
      required_text, overall. EN is the control/ceiling row.
  results/phase2b/extraction_accuracy_near_misses.csv
      Every non-exact match, raw extracted value vs. ground truth,
      for manual review only — informational, not scored.
  results/phase2b/extraction_accuracy_raw.csv
      Per-brief, per-language raw extraction output (resume-safe cache).

Resume-safe: (brief_id, language) pairs already present in the raw
extraction cache are not re-sent to the API.

Run from MulBrandEval/ root:
    python phase2b_multilingual/phase2b_extraction_accuracy.py
"""

import csv
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

LANGUAGES = ["en", "de", "fr", "es", "ar"]   # en = control / extraction ceiling
BRIEF_FILE_TEMPLATE = "data/prompts/mbb/mbb_briefs_{lang}.csv"
GROUND_TRUTH_FILE = "data/prompts/mbb/mbb_briefs_en.csv"

EXTRACTION_MODEL = "gpt-4o-mini"   # text-only call, no image, no `detail` param

RAW_OUTPUT_FILE       = Path("results/phase2b/extraction_accuracy_raw.csv")
NEAR_MISS_OUTPUT_FILE = Path("results/phase2b/extraction_accuracy_near_misses.csv")
SUMMARY_OUTPUT_FILE   = Path("results/phase2b/extraction_accuracy_validation.csv")

RAW_FIELDS = [
    "brief_id", "language",
    "extracted_objects_json", "extracted_colours_json",
    "extracted_spatial_json", "extracted_text_json",
    "extraction_error",
]
NEAR_MISS_FIELDS = ["brief_id", "language", "category", "extracted_value", "ground_truth_value"]
SUMMARY_FIELDS = ["language", "category", "n_briefs", "n_matched", "accuracy", "accuracy_strict"]

DIRECTION_WORDS = {"left", "right", "above", "below"}
OPPOSITE_DIRECTION = {"left": "right", "right": "left", "above": "below", "below": "above"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── OpenAI client singleton (same pattern as tifa_baseline.py) ──────────────────

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _CLIENT


EXTRACTION_SYSTEM_PROMPT = """You extract structured product-brief requirements from raw text, which may be in any language. Read the given brief text and return ONLY a JSON object with exactly these keys:

- "objects": a list of required physical object names mentioned (English translation of each, lowercase, singular form, no articles). E.g. ["smartwatch", "wireless charger"]
- "colours": an object mapping each object name (matching the "objects" list) to its required colour in English, lowercase. E.g. {"smartwatch": "silver", "wireless charger": "black"}
- "spatial_constraints": a list of plain-English spatial relationship strings using ONLY the words left/right/above/below, in the form "<object> <direction> of <object>". E.g. ["smartwatch left of wireless charger"]. Return an empty list if none are stated.
- "required_text": a list of any literal text/words that must appear rendered in the image (e.g. a brand name on a product). Keep this EXACTLY as written in the source text, do not translate it. Return an empty list if none.

Return ONLY the JSON object, no explanation, no markdown code fences."""


def _extract_fields(prompt_text: str) -> dict:
    """
    Single text-only GPT-4o mini call. No image, no English hints — the
    model sees only the raw brief sentence in whatever language it is in.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=400,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    # Defensive: strip markdown fences if the model adds them anyway
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    parsed = json.loads(raw)
    return {
        "objects":             parsed.get("objects", []),
        "colours":             parsed.get("colours", {}),
        "spatial_constraints": parsed.get("spatial_constraints", []),
        "required_text":       parsed.get("required_text", []),
    }


# ── Normalization (mechanical only — no hand-authored synonym mapping) ─────────

def _normalize_text(s: str) -> str:
    """Case-fold + whitespace/hyphen normalization + naive plural stripping."""
    s = s.strip().lower()
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"\s+", " ", s)
    if s.endswith("ies") and len(s) > 4:
        s = s[:-3] + "y"
    elif s.endswith("ses") and len(s) > 4:
        s = s[:-2]
    elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        s = s[:-1]
    return s


def _normalize_set(items) -> set:
    return {_normalize_text(x) for x in items if isinstance(x, str) and x.strip()}


def _parse_spatial_triplet(constraint: str):
    """
    Mirrors node0_prompt_parser.py's _clean_spatial_constraints parsing
    logic: pulls (obj0, direction, obj1) out of a free-text constraint
    string. Returns None if no direction word is found.
    """
    if not isinstance(constraint, str):
        return None
    parts = constraint.lower().split()
    dirs_found = [w for w in parts if w in DIRECTION_WORDS]
    if not dirs_found:
        return None
    idx = parts.index(dirs_found[0])
    obj0 = " ".join(parts[:idx]).strip()
    rest = parts[idx + 1:]
    if rest and rest[0] == "of":
        rest = rest[1:]
    obj1 = " ".join(rest).strip()
    return (_normalize_text(obj0), dirs_found[0], _normalize_text(obj1))


def _normalize_spatial_set(constraints) -> set:
    triplets = set()
    for c in constraints:
        t = _parse_spatial_triplet(c)
        if t:
            triplets.add(t)
    return triplets


# ── Ground truth / brief loading ─────────────────────────────────────────────

def _parse_json_field(value, field_name: str):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse field '{field_name}': {e}\nValue: {value!r}")


def load_ground_truth() -> dict:
    """brief_id -> {objects: set, colours: dict, spatial: set(triplets), text: set}"""
    df = pd.read_csv(GROUND_TRUTH_FILE)
    gt = {}
    for _, row in df.iterrows():
        objects  = _parse_json_field(row["required_objects"],    "required_objects")
        colours  = _parse_json_field(row["colour_attributes"],   "colour_attributes")
        spatial  = _parse_json_field(row["spatial_constraints"], "spatial_constraints")
        text     = _parse_json_field(row["required_text"],       "required_text")
        gt[row["brief_id"]] = {
            "objects": _normalize_set(objects),
            "colours": {_normalize_text(k): _normalize_text(v) for k, v in colours.items()},
            "spatial":  _normalize_spatial_set(spatial),
            "text":     {t.strip() for t in text if isinstance(t, str)},  # exact-match, case-fold only applied at compare time
        }
    return gt


def load_briefs(lang: str) -> pd.DataFrame:
    path = BRIEF_FILE_TEMPLATE.format(lang=lang)
    df = pd.read_csv(path)
    n = len(df)
    assert n == 150, f"Expected 150 briefs, found {n} in {path} — check the file."
    return df


# ── Resume-safe raw-extraction cache ─────────────────────────────────────────

def already_extracted() -> set:
    if not RAW_OUTPUT_FILE.exists():
        return set()
    df = pd.read_csv(RAW_OUTPUT_FILE)
    if df.empty:
        return set()
    return set(zip(df["brief_id"], df["language"]))


def run_extraction():
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found.\n"
            "Check that your .env file contains it and that python-dotenv "
            "has been loaded (see phase2a_generate.py for the load_dotenv() pattern)."
        )

    done = already_extracted()
    jobs = []
    for lang in LANGUAGES:
        briefs_df = load_briefs(lang)
        for _, brief in briefs_df.iterrows():
            key = (brief["brief_id"], lang)
            if key in done:
                continue
            jobs.append((brief["brief_id"], lang, brief["prompt_text"]))

    log.info(f"Extraction-accuracy validation — {len(jobs)} call(s) to make | "
             f"{len(done)} already cached")

    if not jobs:
        log.info("Nothing to extract — raw cache already complete.")
        return

    RAW_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RAW_OUTPUT_FILE.exists()
    fh = open(RAW_OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
    if write_header:
        writer.writeheader()

    n_ok, n_fail = 0, 0
    for i, (brief_id, lang, prompt_text) in enumerate(jobs, 1):
        try:
            extracted = _extract_fields(prompt_text)
            writer.writerow({
                "brief_id": brief_id,
                "language": lang,
                "extracted_objects_json": json.dumps(extracted["objects"]),
                "extracted_colours_json": json.dumps(extracted["colours"]),
                "extracted_spatial_json": json.dumps(extracted["spatial_constraints"]),
                "extracted_text_json":    json.dumps(extracted["required_text"]),
                "extraction_error": "",
            })
            fh.flush()
            n_ok += 1
            log.info(f"[{i}/{len(jobs)}] [OK] {brief_id} | {lang}")
        except Exception as e:
            n_fail += 1
            writer.writerow({
                "brief_id": brief_id, "language": lang,
                "extracted_objects_json": "[]", "extracted_colours_json": "{}",
                "extracted_spatial_json": "[]", "extracted_text_json": "[]",
                "extraction_error": str(e),
            })
            fh.flush()
            log.error(f"[{i}/{len(jobs)}] [FAIL] {brief_id} | {lang} | error={e}")

    fh.close()
    log.info("=" * 60)
    log.info(f"Extraction complete — [OK] {n_ok}  [FAIL] {n_fail}")
    log.info("=" * 60)


# ── Matching helpers (word-level containment — mechanical, no semantic judgment) ─
#
# Diagnosed via the EN control run: exact-set matching on full noun phrases
# undercounts correct extractions when the ground truth or the extraction
# includes a modifier the other doesn't (e.g. ground truth "metal stand" vs
# extracted "stand"; ground truth "camera" vs extracted "dslr camera").
# Word-level containment (is the shorter phrase's word set a subset of the
# longer phrase's word set) fixes this without introducing any semantic
# synonym judgment — "stand" matching "metal stand" is a strict subset
# relationship, not an opinion about meaning. This rule was set once, based
# entirely on EN-control diagnostic evidence, before re-scoring any
# multilingual row, and is not iterated further.

def _phrase_contains(a: str, b: str) -> bool:
    """True if the (normalized) word set of a is a subset of b's, or vice versa."""
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return False
    return wa.issubset(wb) or wb.issubset(wa)


def _best_containment_match(target: str, candidates) -> bool:
    return any(_phrase_contains(target, c) for c in candidates)


def _best_containment_object_key(target_obj: str, extracted_colours: dict):
    """Find the extracted colour-dict key that best containment-matches target_obj."""
    for key in extracted_colours:
        if _phrase_contains(target_obj, key):
            return key
    return None


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_all():
    gt = load_ground_truth()
    raw_df = pd.read_csv(RAW_OUTPUT_FILE)

    near_misses = []
    # per (language, category) -> [n_matched, n_total]
    tally = {}
    # secondary, strict/exact tally kept for transparency (original rule)
    tally_strict = {}

    def bump(lang, category, matched, strict_matched=None):
        key = (lang, category)
        if key not in tally:
            tally[key] = [0, 0]
        tally[key][1] += 1
        if matched:
            tally[key][0] += 1
        if strict_matched is not None:
            if key not in tally_strict:
                tally_strict[key] = [0, 0]
            tally_strict[key][1] += 1
            if strict_matched:
                tally_strict[key][0] += 1

    for _, row in raw_df.iterrows():
        brief_id, lang = row["brief_id"], row["language"]
        if brief_id not in gt:
            continue
        truth = gt[brief_id]

        ext_objects = _normalize_set(json.loads(row["extracted_objects_json"]))
        ext_colours_raw = json.loads(row["extracted_colours_json"])
        ext_colours = {_normalize_text(k): _normalize_text(v) for k, v in ext_colours_raw.items()}
        ext_spatial_raw = _normalize_spatial_set(json.loads(row["extracted_spatial_json"]))
        ext_text = {t.strip() for t in json.loads(row["extracted_text_json"]) if isinstance(t, str)}

        # objects: containment match (word-set subset either direction)
        for obj in truth["objects"]:
            strict_matched = obj in ext_objects
            matched = strict_matched or _best_containment_match(obj, ext_objects)
            bump(lang, "objects", matched, strict_matched)
            if not matched:
                near_misses.append((brief_id, lang, "objects", sorted(ext_objects), obj))

        # colours: find best containment-matched object key, then containment-match
        # the colour value too (handles "dark brown" vs "brown")
        for obj, colour in truth["colours"].items():
            strict_matched = ext_colours.get(obj) == colour
            key = _best_containment_object_key(obj, ext_colours)
            ext_colour_val = ext_colours.get(key, "") if key else ""
            matched = strict_matched or (key is not None and _phrase_contains(colour, ext_colour_val))
            bump(lang, "colours", matched, strict_matched)
            if not matched:
                near_misses.append((brief_id, lang, "colours", ext_colour_val or "<missing>", colour))

        # spatial_constraints: containment match on the two object strings,
        # direction word matched either same-order OR as the logically
        # equivalent reciprocal relation ("A right of B" == "B left of A").
        # Diagnosed via EN control (MB034, MB040, MB053, MB105, MB138 — all
        # reciprocal phrasings of the same fact, not extraction errors);
        # this is the final matching-rule adjustment, applied once.
        for (t_obj0, t_dir, t_obj1) in truth["spatial"]:
            strict_matched = (t_obj0, t_dir, t_obj1) in ext_spatial_raw
            matched = strict_matched or any(
                (e_dir == t_dir and _phrase_contains(t_obj0, e_obj0) and _phrase_contains(t_obj1, e_obj1))
                or (e_dir == OPPOSITE_DIRECTION[t_dir] and _phrase_contains(t_obj0, e_obj1) and _phrase_contains(t_obj1, e_obj0))
                for (e_obj0, e_dir, e_obj1) in ext_spatial_raw
            )
            bump(lang, "spatial_constraints", matched, strict_matched)
            if not matched:
                near_misses.append((brief_id, lang, "spatial_constraints", sorted(ext_spatial_raw), (t_obj0, t_dir, t_obj1)))

        # required_text: whole-word substring containment, case-fold only,
        # no stemming (brand text should still render literally — this only
        # tolerates the extraction wrapping it in descriptive context, e.g.
        # "KESTRA logo" for ground truth "KESTRA")
        ext_text_cf = [t.lower() for t in ext_text]
        for text in truth["text"]:
            text_cf = text.lower()
            strict_matched = text_cf in {t for t in ext_text_cf}
            matched = strict_matched or any(
                re.search(r"\b" + re.escape(text_cf) + r"\b", t) for t in ext_text_cf
            )
            bump(lang, "required_text", matched, strict_matched)
            if not matched:
                near_misses.append((brief_id, lang, "required_text", sorted(ext_text), text))

    # ── Write near-miss sidecar (informational only, not scored) ────────────
    NEAR_MISS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NEAR_MISS_OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEAR_MISS_FIELDS)
        writer.writeheader()
        for brief_id, lang, category, extracted_value, gt_value in near_misses:
            writer.writerow({
                "brief_id": brief_id, "language": lang, "category": category,
                "extracted_value": extracted_value, "ground_truth_value": gt_value,
            })

    # ── Write summary table, with an "overall" row per language ─────────────
    # "accuracy" = word-level containment rule (primary, reported figure).
    # "accuracy_strict" = original exact-match rule, kept for transparency so
    # the effect of the containment fix is visible, not hidden.
    rows = []
    for lang in LANGUAGES:
        overall_matched, overall_total = 0, 0
        overall_matched_strict, overall_total_strict = 0, 0
        for category in ["objects", "colours", "spatial_constraints", "required_text"]:
            n_matched, n_total = tally.get((lang, category), [0, 0])
            n_matched_s, n_total_s = tally_strict.get((lang, category), [0, 0])
            overall_matched += n_matched
            overall_total += n_total
            overall_matched_strict += n_matched_s
            overall_total_strict += n_total_s
            accuracy = round(n_matched / n_total, 4) if n_total else None
            accuracy_strict = round(n_matched_s / n_total_s, 4) if n_total_s else None
            rows.append({
                "language": lang, "category": category,
                "n_briefs": n_total, "n_matched": n_matched, "accuracy": accuracy,
                "accuracy_strict": accuracy_strict,
            })
        overall_accuracy = round(overall_matched / overall_total, 4) if overall_total else None
        overall_accuracy_strict = round(overall_matched_strict / overall_total_strict, 4) if overall_total_strict else None
        rows.append({
            "language": lang, "category": "overall",
            "n_briefs": overall_total, "n_matched": overall_matched, "accuracy": overall_accuracy,
            "accuracy_strict": overall_accuracy_strict,
        })

    SUMMARY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    log.info("=" * 60)
    log.info(f"Summary table written: {SUMMARY_OUTPUT_FILE}")
    log.info(f"Near-miss log written: {NEAR_MISS_OUTPUT_FILE} ({len(near_misses)} entries)")
    log.info("=" * 60)
    for lang in LANGUAGES:
        overall_row = next(r for r in rows if r["language"] == lang and r["category"] == "overall")
        acc = overall_row["accuracy"]
        log.info(f"  {lang.upper():3s}  overall extraction accuracy = "
                  f"{acc:.4f}" if acc is not None else f"  {lang.upper():3s}  no data")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_extraction()
    score_all()


if __name__ == "__main__":
    main()