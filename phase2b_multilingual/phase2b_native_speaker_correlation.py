"""
phase2b_multilingual/phase2b_native_speaker_correlation.py

Layer 3 corroboration check: correlates native-speaker Likert ratings of
translated brand briefs against the two automated translation-quality
metrics already computed for every brief -- cc_score (mCLIP cycle-
consistency) and cometkiwi_score (Unbabel/wmt22-cometkiwi-da) -- to see
whether the automated metrics track what a human perceives as translation
quality. This is supplementary corroboration, not a standalone inferential
claim: the automated metrics remain the primary translation-quality gate
used everywhere else in this thesis (e.g. the 37 REVIEW/FAIL dispositions).

Each of the 10 sampled briefs per language was rated on two dimensions --
fluency and semantic fidelity -- by every eligible native-speaker
respondent. Ratings are averaged across raters per brief per dimension,
then correlated (Spearman, primary; Pearson, secondary) against cc_score
and cometkiwi_score, both per language and pooled across all four.

Respondents who selected "None of these" for native language are excluded
(the form correctly skips them past every rating section, so they have no
data to exclude in practice, but the check is explicit here regardless).

Run from project root:
    python phase2b_multilingual/phase2b_native_speaker_correlation.py
"""
import pandas as pd
from scipy import stats

RESPONSES_PATH = "results/phase2b/phase2b_native_speaker_survey_responses.csv"
OUTPUT_PATH = "results/phase2b/phase2b_native_speaker_correlation_results.csv"

BRIEFS_PATH_TEMPLATE = "data/prompts/mbb/mbb_briefs_{lang}.csv"

NATIVE_LANGUAGE_COL = "What is your native language (the language you learned first as a child)?"

# Column-suffix pattern per language: (native_language_value, fluency_suffix, semantic_fidelity_suffix)
LANGUAGE_CONFIG = [
    ("German", "[SF]", "[BT]"),
    ("French", "[Fl]", "[FS]"),
    ("Spanish", "[Fl]", "[FS]"),
    ("Arabic", "[الطلاقة]", "[الدقة]"),
]

LANG_CODE = {"German": "de", "French": "fr", "Spanish": "es", "Arabic": "ar"}


def extract_language_ratings(df: pd.DataFrame, native_language: str, fl_suffix: str, sf_suffix: str) -> pd.DataFrame:
    """Long-format brief-level mean ratings for one language's respondents."""
    sub = df[df[NATIVE_LANGUAGE_COL] == native_language]
    if len(sub) == 0:
        return pd.DataFrame(columns=["brief_id", "fluency", "semantic_fidelity", "n_raters"])

    fl_cols = [c for c in sub.columns if c.endswith(fl_suffix) and c.startswith("MB")]
    sf_cols = [c for c in sub.columns if c.endswith(sf_suffix) and c.startswith("MB")]
    brief_ids = sorted({c.split(" ")[0] for c in fl_cols})

    rows = []
    for bid in brief_ids:
        fl_col = f"{bid} {fl_suffix}"
        sf_col = f"{bid} {sf_suffix}"
        fl_vals = sub[fl_col].dropna()
        sf_vals = sub[sf_col].dropna()
        if len(fl_vals) == 0 or len(sf_vals) == 0:
            continue
        rows.append({
            "brief_id": bid,
            "fluency": fl_vals.mean(),
            "semantic_fidelity": sf_vals.mean(),
            "n_raters": len(fl_vals),
        })
    return pd.DataFrame(rows)


def correlate(df: pd.DataFrame, human_col: str, metric_col: str, label: str, dimension: str) -> dict:
    sub = df.dropna(subset=[human_col, metric_col])
    if len(sub) < 4:
        return {
            "language": label, "dimension": dimension, "metric": metric_col,
            "n": len(sub), "spearman_rho": None, "spearman_p": None,
            "pearson_r": None, "pearson_p": None,
        }
    sp = stats.spearmanr(sub[human_col], sub[metric_col])
    pe = stats.pearsonr(sub[human_col], sub[metric_col])
    return {
        "language": label, "dimension": dimension, "metric": metric_col,
        "n": len(sub), "spearman_rho": sp.correlation, "spearman_p": sp.pvalue,
        "pearson_r": pe[0], "pearson_p": pe[1],
    }


def main():
    df = pd.read_csv(RESPONSES_PATH)

    n_total = len(df)
    n_eligible = (df[NATIVE_LANGUAGE_COL] != "None of these").sum()
    print(f"Total responses: {n_total} | Eligible (native-language match): {n_eligible}")
    print(f"Breakdown: {df[NATIVE_LANGUAGE_COL].value_counts().to_dict()}\n")

    all_ratings = []
    results = []

    for native_lang, fl_suffix, sf_suffix in LANGUAGE_CONFIG:
        lang_code = LANG_CODE[native_lang]
        ratings = extract_language_ratings(df, native_lang, fl_suffix, sf_suffix)
        if len(ratings) == 0:
            print(f"{native_lang}: no responses yet, skipping.")
            continue

        ratings["human_overall"] = (ratings["fluency"] + ratings["semantic_fidelity"]) / 2
        ratings["language"] = lang_code

        briefs = pd.read_csv(BRIEFS_PATH_TEMPLATE.format(lang=lang_code))
        merged = ratings.merge(briefs[["brief_id", "cc_score", "cometkiwi_score"]], on="brief_id", how="left")

        n_missing = merged["cc_score"].isna().sum()
        if n_missing > 0:
            print(f"WARNING [{native_lang}]: {n_missing} brief_id(s) had no match in {BRIEFS_PATH_TEMPLATE.format(lang=lang_code)} -- check item IDs.")

        all_ratings.append(merged)

        n_raters = ratings["n_raters"].max()
        print(f"{native_lang}: {len(merged)} items, up to {n_raters} rater(s)/item")

        for metric_col in ["cc_score", "cometkiwi_score"]:
            for dim_col, dim_label in [("human_overall", "overall"), ("fluency", "fluency"), ("semantic_fidelity", "semantic_fidelity")]:
                result = correlate(merged, dim_col, metric_col, native_lang, dim_label)
                results.append(result)
                if dim_label == "overall":
                    rho = result["spearman_rho"]
                    p = result["spearman_p"]
                    rho_str = f"{rho:.4f}" if rho is not None else "n/a (n<4)"
                    p_str = f"{p:.4f}" if p is not None else "n/a"
                    print(f"  vs {metric_col}: Spearman rho={rho_str}, p={p_str}, n={result['n']}")
        print()

    if all_ratings:
        pooled = pd.concat(all_ratings, ignore_index=True)
        print(f"Pooled across all languages: {len(pooled)} items\n")
        for metric_col in ["cc_score", "cometkiwi_score"]:
            for dim_col, dim_label in [("human_overall", "overall"), ("fluency", "fluency"), ("semantic_fidelity", "semantic_fidelity")]:
                result = correlate(pooled, dim_col, metric_col, "Pooled (all languages)", dim_label)
                results.append(result)
                if dim_label == "overall":
                    rho = result["spearman_rho"]
                    p = result["spearman_p"]
                    rho_str = f"{rho:.4f}" if rho is not None else "n/a (n<4)"
                    p_str = f"{p:.4f}" if p is not None else "n/a"
                    print(f"  vs {metric_col}: Spearman rho={rho_str}, p={p_str}, n={result['n']}")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()