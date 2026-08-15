"""
full_brief_validation.py

Broader pre-generation check, run once across all 150 briefs, covering
every structured field the DAG actually evaluates against — not just
spatial direction (which is the only class of bug check_spatial_v2.py
looks for).

Four checks:

  1. SPATIAL DIRECTION — reuses the object-order-aware logic from
     check_spatial_v2.py (prompt vs spatial_constraints).

  2. COLOUR PRESENCE — for each object's assigned colour in
     colour_attributes, checks that colour word (or a close synonym from
     a small manual map) appears somewhere in prompt_text. Flags misses
     for a human glance — colour naming is genuinely fuzzy (teal vs
     blue-green vs turquoise all plausibly refer to the same brief), so
     this check is intentionally permissive and only flags cases with NO
     match at all, not stylistic mismatches.

  3. REQUIRED_TEXT PRESENCE — verifies the exact brand-name string in
     required_text appears in prompt_text (this is what Node 4 looks for
     in the generated image, so if it's not even in the prompt, Node 4
     cannot possibly pass).

  4. CROSS-LANGUAGE FIELD INTEGRITY — for spatial_constraints,
     colour_attributes, required_objects, required_text, style_descriptors
     (the five fields that must NEVER be translated), confirms each
     translated file (DE/FR/ES/AR) has byte-identical values to the
     English file for every brief. This catches copy/write corruption in
     phase2b_translate.py's output, independent of anything DeepL did.

Usage:
    python full_brief_validation.py \\
        --en data/prompts/mbb/mbb_briefs_en.csv \\
        --de data/prompts/mbb/mbb_briefs_de.csv \\
        --fr data/prompts/mbb/mbb_briefs_fr.csv \\
        --es data/prompts/mbb/mbb_briefs_es.csv \\
        --ar data/prompts/mbb/mbb_briefs_ar.csv
"""

import argparse
import json
import re
import pandas as pd

# ── Shared direction logic (same as check_spatial_v2.py) ───────────────────
DIRECTION_PATTERNS = [
    ("left_of",  re.compile(r"\bleft of\b", re.IGNORECASE)),
    ("right_of", re.compile(r"\bright of\b", re.IGNORECASE)),
    ("above",    re.compile(r"\babove\b|\bon top of\b|\bmounted above\b|\bresting (on|atop)\b|\bplaced on\b", re.IGNORECASE)),
    ("below",    re.compile(r"\bbelow\b|\bbeneath\b|\bunder(?:neath)?\b", re.IGNORECASE)),
]
OPPOSITES = {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above"}

# Small manual synonym map for the colour-presence check — extend as needed.
COLOUR_SYNONYMS = {
    "teal": ["teal", "blue-green", "blue green", "turquoise"],
    "tan": ["tan", "light brown", "beige"],
    "gold": ["gold", "golden"],
    "navy": ["navy", "dark blue"],
    "rose gold": ["rose gold", "pink gold"],
}


def normalize_direction(text):
    if not isinstance(text, str):
        return None
    for label, pattern in DIRECTION_PATTERNS:
        if pattern.search(text):
            return label
    return None


def parse_json_list(raw):
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [s.strip(' "\'') for s in raw.strip("[]").split(",") if s.strip(' "\'')]


def parse_json_dict(raw):
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def parse_constraint_triple(constraint, obj_a, obj_b):
    lower = constraint.lower()
    idx_a, idx_b = lower.find(obj_a.lower()), lower.find(obj_b.lower())
    if idx_a == -1 or idx_b == -1:
        return None
    direction = normalize_direction(constraint)
    if direction is None:
        return None
    return (obj_a, direction, obj_b) if idx_a < idx_b else (obj_b, direction, obj_a)


def extract_prompt_triple(prompt, obj_a, obj_b):
    lower = prompt.lower()
    idx_a, idx_b = lower.find(obj_a.lower()), lower.find(obj_b.lower())
    if idx_a == -1 or idx_b == -1:
        return None
    if idx_a < idx_b:
        first, first_idx, second, second_idx = obj_a, idx_a, obj_b, idx_b
    else:
        first, first_idx, second, second_idx = obj_b, idx_b, obj_a, idx_a
    between      = lower[first_idx + len(first): second_idx]
    after_second = lower[second_idx + len(second): second_idx + len(second) + 60]
    dir1 = normalize_direction(between)
    if dir1:
        return (first, dir1, second)
    dir2 = normalize_direction(after_second)
    if dir2 and re.search(r"\bits\b|\bit\b", after_second[:30]):
        return (second, dir2, first)
    return None


def compare_triples(c_triple, p_triple):
    c_subj, c_dir, c_ref = c_triple
    p_subj, p_dir, p_ref = p_triple
    if c_subj == p_subj and c_ref == p_ref:
        return c_dir == p_dir
    elif c_subj == p_ref and c_ref == p_subj:
        return OPPOSITES.get(c_dir) == p_dir
    return None


# ── Check 1: spatial direction ──────────────────────────────────────────────
def check_spatial(df, prompt_col):
    contradictions = []
    for _, row in df.iterrows():
        objs = parse_json_list(row["required_objects"])
        constraints = parse_json_list(row["spatial_constraints"])
        if len(objs) != 2 or not constraints:
            continue
        obj_a, obj_b = objs[0], objs[1]
        for constraint in constraints:
            c_triple = parse_constraint_triple(constraint, obj_a, obj_b)
            p_triple = extract_prompt_triple(row[prompt_col], obj_a, obj_b)
            if c_triple is None or p_triple is None:
                continue
            if compare_triples(c_triple, p_triple) is False:
                contradictions.append(row["brief_id"])
    return contradictions


# ── Check 2: colour presence ────────────────────────────────────────────────
def check_colour_presence(df, prompt_col):
    misses = []
    for _, row in df.iterrows():
        colours = parse_json_dict(row["colour_attributes"])
        prompt_lower = row[prompt_col].lower()
        for obj, colour in colours.items():
            candidates = COLOUR_SYNONYMS.get(colour.lower(), [colour.lower()])
            if not any(c in prompt_lower for c in candidates):
                misses.append((row["brief_id"], obj, colour))
    return misses


# ── Check 3: required_text presence ─────────────────────────────────────────
def check_required_text_presence(df, prompt_col):
    misses = []
    for _, row in df.iterrows():
        texts = parse_json_list(row["required_text"])
        prompt_lower = row[prompt_col].lower()
        for t in texts:
            if t.lower() not in prompt_lower:
                misses.append((row["brief_id"], t))
    return misses


# ── Check 4: cross-language field integrity ─────────────────────────────────
LOCKED_FIELDS = ["required_objects", "colour_attributes", "spatial_constraints",
                  "required_text", "style_descriptors"]


def check_cross_language(df_en, df_lang, lang_tag):
    mismatches = []
    en_indexed = df_en.set_index("brief_id")
    lang_indexed = df_lang.set_index("brief_id")

    for brief_id in en_indexed.index:
        if brief_id not in lang_indexed.index:
            mismatches.append((brief_id, lang_tag, "MISSING_ROW", None, None))
            continue
        for field in LOCKED_FIELDS:
            en_val = str(en_indexed.loc[brief_id, field]).strip()
            lang_val = str(lang_indexed.loc[brief_id, field]).strip()
            if en_val != lang_val:
                mismatches.append((brief_id, lang_tag, field, en_val, lang_val))
    return mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", required=True)
    ap.add_argument("--de")
    ap.add_argument("--fr")
    ap.add_argument("--es")
    ap.add_argument("--ar")
    args = ap.parse_args()

    df_en = pd.read_csv(args.en)
    prompt_col_en = "prompt_text_en" if "prompt_text_en" in df_en.columns else "prompt_text"

    print("=" * 70)
    print(f"CHECK 1 — Spatial direction consistency ({len(df_en)} briefs)")
    print("=" * 70)
    contradictions = check_spatial(df_en, prompt_col_en)
    if contradictions:
        print(f"CONTRADICTIONS FOUND: {contradictions}")
    else:
        print("None found.")

    print("\n" + "=" * 70)
    print("CHECK 2 — Colour presence in prompt_text")
    print("=" * 70)
    colour_misses = check_colour_presence(df_en, prompt_col_en)
    if colour_misses:
        print(f"{len(colour_misses)} case(s) with no matching colour word found "
              f"(may be fine — extend COLOUR_SYNONYMS if these are just unmapped synonyms):")
        for bid, obj, colour in colour_misses:
            print(f"  {bid}: {obj} = '{colour}' — not found in prompt")
    else:
        print("All colours found in their briefs' prompt text.")

    print("\n" + "=" * 70)
    print("CHECK 3 — required_text presence in prompt_text")
    print("=" * 70)
    text_misses = check_required_text_presence(df_en, prompt_col_en)
    if text_misses:
        print(f"{len(text_misses)} case(s) — required_text NOT found in prompt_text "
              f"(these WILL fail Node 4 regardless of image quality):")
        for bid, t in text_misses:
            print(f"  {bid}: '{t}' not in prompt")
    else:
        print("All required_text values found in their briefs' prompt text.")

    print("\n" + "=" * 70)
    print("CHECK 4 — Cross-language field integrity (locked fields must be byte-identical)")
    print("=" * 70)
    lang_files = {"de": args.de, "fr": args.fr, "es": args.es, "ar": args.ar}
    any_mismatch = False
    for lang, path in lang_files.items():
        if not path:
            continue
        df_lang = pd.read_csv(path)
        mismatches = check_cross_language(df_en, df_lang, lang)
        if mismatches:
            any_mismatch = True
            print(f"\n{lang.upper()}: {len(mismatches)} mismatch(es)")
            for bid, lg, field, en_val, lang_val in mismatches:
                print(f"  {bid} [{field}]:")
                print(f"    EN : {en_val}")
                print(f"    {lg.upper()} : {lang_val}")
        else:
            print(f"{lang.upper()}: all {len(LOCKED_FIELDS)} locked fields identical across "
                  f"{len(df_lang)} briefs. Clean.")

    if not any_mismatch:
        print("\nAll translated files carry the locked fields unchanged from English. Clean.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Spatial contradictions : {len(contradictions)}")
    print(f"Colour presence misses : {len(colour_misses)}")
    print(f"required_text misses   : {len(text_misses)}")
    print(f"Cross-language issues  : {'yes — see above' if any_mismatch else 0}")


if __name__ == "__main__":
    main()