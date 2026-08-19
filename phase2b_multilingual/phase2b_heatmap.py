"""
MulBrandEval — Phase 2B Per-Node CLCG Heatmap
==============================================
Generates two per-node CLCG heatmaps and their precision-companion CSV:

  Heatmap 1 — Configuration B (masked):
      The operational DAG pipeline with short-circuit logic.
      Short-circuited images (Node 1 fail → skip Nodes 2–5) are
      zero-imputed by the coordinator, so Node 2/3 means include
      hard 0.0s for unmeasured images. This is what the deployed
      pipeline actually reports.

  Heatmap 2 — Configuration C (unmasked):
      Same DAG nodes, unconditional edges (no short-circuit).
      Every node runs on every image. Represents the true per-node
      degradation signal, uncontaminated by zero-imputation.

Both heatmaps share the same diverging color scale (RdBu_r, centered
at 0), so they can be directly compared visually. Negative CLCG
(language scores ABOVE English — e.g. FLUX Node 4 in DE/FR/ES) is
shown in blue; positive CLCG (language scores below English) in red.

CLCG formula (consistent with Phase 1A/1B convention):
    CLCG(node, lang, model) = mean_score(EN) − mean_score(lang)

N/A cells: SD v1.5 Node 4 (architecturally undefined — SD v1.5
cannot render text reliably; Node 4 CLCG would measure encoder
noise, not compliance). Driven by node4_clcg_interpretable flag.

Outputs (all to results/phase2b/):
    heatmap_config_b_masked.png    — Config B standalone (2 model panels)
    heatmap_config_c_unmasked.png  — Config C standalone (2 model panels)
    heatmap_combined.png           — 2×2 combined figure for thesis
    per_node_clcg_heatmap.csv      — 30-row precision companion CSV

Run from MulBrandEval/ project root:
    python phase2b_multilingual/phase2b_heatmap.py
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Input files ───────────────────────────────────────────────────────────────

B_MULTILINGUAL = Path("results/phase2b/multilingual_ablation_table.csv")
C_MULTILINGUAL = Path("results/phase2b/configuration_c_flat_pipeline_multilingual.csv")
B_ENGLISH      = Path("results/phase2a/english_ablation_table.csv")
C_ENGLISH      = Path("results/phase2a/configuration_c_flat_pipeline.csv")

OUTPUT_DIR = Path("results/phase2b")

# ── Constants ─────────────────────────────────────────────────────────────────

LANGUAGES   = ["de", "fr", "es"]
LANG_LABELS = {"de": "DE", "fr": "FR", "es": "ES"}

MODELS       = ["sd15", "flux"]
MODEL_LABELS = {"sd15": "SD v1.5", "flux": "FLUX.1-dev"}

NODES = ["node1", "node2", "node3", "node4", "node5"]
NODE_LABELS = {
    "node1": "Node 1\nObject Presence",
    "node2": "Node 2\nSpatial Layout",
    "node3": "Node 3\nColour Attribute",
    "node4": "Node 4\nTypography",
    "node5": "Node 5\nBrand Quality",
}

# Cells that are architecturally undefined — rendered as N/A
# SD v1.5 Node 4: text-rendering architectural limitation, not a language finding.
# Driven by node4_clcg_interpretable == False in the evaluation data.
NA_CELLS = {("node4", "sd15")}

# ── CLCG computation ──────────────────────────────────────────────────────────

def compute_grid(df_en: pd.DataFrame, df_ml: pd.DataFrame, config_label: str) -> pd.DataFrame:
    """
    Compute per-node CLCG for one configuration.

    Returns a DataFrame with one row per (node × language × model) combination
    — 30 rows total (5 nodes × 3 languages × 2 models).

    N/A cells (SD v1.5 Node 4) have clcg=NaN and interpretable=False.

    n_images reflects how many images actually contributed a real measurement
    to the language mean (relevant for Node 2/3/4 in Config B where
    short-circuited images are zero-imputed rather than omitted).
    """
    records = []
    for model in MODELS:
        en_rows   = df_en[df_en["model"] == model]
        for lang in LANGUAGES:
            lang_rows = df_ml[
                (df_ml["model"] == model) & (df_ml["language"] == lang)
            ]
            for node in NODES:
                score_col    = f"{node}_score"
                en_mean      = en_rows[score_col].mean()
                lang_mean    = lang_rows[score_col].mean()
                interpretable = (node, model) not in NA_CELLS

                clcg = round(en_mean - lang_mean, 4) if interpretable else None

                # n_images: how many images were actually scored (not zero-imputed)
                # Nodes 2, 3, 4 have a skipped flag; Nodes 1 and 5 always run.
                skipped_col = f"{node}_skipped"
                if skipped_col in lang_rows.columns:
                    n = int((lang_rows[skipped_col] == False).sum())
                else:
                    n = len(lang_rows)

                records.append({
                    "config":        config_label,
                    "node":          node,
                    "language":      lang,
                    "model":         model,
                    "en_score":      round(en_mean, 4),
                    "lang_score":    round(lang_mean, 4),
                    "clcg":          clcg,
                    "n_images":      n,
                    "interpretable": interpretable,
                })
    return pd.DataFrame(records)


# ── Heatmap drawing ───────────────────────────────────────────────────────────

NA_COLOR   = "#c8c8c8"   # grey for N/A cells
NA_TEXT_COLOR = "#555555"

def draw_panel(
    ax: plt.Axes,
    grid_df: pd.DataFrame,
    model: str,
    panel_title: str,
    vmin: float,
    vmax: float,
    cmap,
    norm,
    show_ylabel: bool = True,
) -> plt.cm.ScalarMappable:
    """
    Draw one 5-node × 3-language heatmap panel onto ax.
    Returns the ScalarMappable for the shared colorbar.
    """
    n_nodes = len(NODES)
    n_langs = len(LANGUAGES)

    matrix   = np.full((n_nodes, n_langs), np.nan)
    na_mask  = np.zeros((n_nodes, n_langs), dtype=bool)

    for j, lang in enumerate(LANGUAGES):
        for i, node in enumerate(NODES):
            row = grid_df[
                (grid_df["model"]    == model) &
                (grid_df["language"] == lang)  &
                (grid_df["node"]     == node)
            ]
            if row.empty:
                continue
            if not row.iloc[0]["interpretable"]:
                na_mask[i, j] = True
            else:
                val = row.iloc[0]["clcg"]
                if val is not None and not np.isnan(val):
                    matrix[i, j] = val

    # --- Draw heatmap cells ---
    masked_matrix = np.ma.masked_where(np.isnan(matrix) | na_mask, matrix)
    sm = ax.imshow(masked_matrix, cmap=cmap, norm=norm, aspect="auto")

    # --- Draw N/A cells with grey background ---
    na_display = np.ma.masked_where(~na_mask, np.zeros_like(matrix))
    na_cmap = mcolors.ListedColormap([NA_COLOR])
    ax.imshow(na_display, cmap=na_cmap, vmin=0, vmax=1, aspect="auto")

    # --- Annotations ---
    for i in range(n_nodes):
        for j in range(n_langs):
            if na_mask[i, j]:
                ax.text(
                    j, i, "N/A",
                    ha="center", va="center",
                    fontsize=8.5, color=NA_TEXT_COLOR, fontweight="bold",
                )
            elif not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                # White text on saturated cells, black on pale cells
                normed = norm(val)
                r, g, b, _ = cmap(normed)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if luminance < 0.45 else "black"
                ax.text(
                    j, i, f"{val:+.3f}",
                    ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold",
                )

    # --- Axes labels ---
    ax.set_xticks(range(n_langs))
    ax.set_xticklabels(
        [LANG_LABELS[l] for l in LANGUAGES], fontsize=10, fontweight="bold"
    )
    if show_ylabel:
        ax.set_yticks(range(n_nodes))
        ax.set_yticklabels(
            [NODE_LABELS[n] for n in NODES], fontsize=9
        )
    else:
        ax.set_yticks(range(n_nodes))
        ax.set_yticklabels([])

    ax.set_title(
        panel_title,
        fontsize=10, fontweight="bold", pad=8,
    )

    # --- Grid lines ---
    ax.set_xticks(np.arange(-0.5, n_langs, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_nodes, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.tick_params(which="major", length=0)

    return sm


def save_single_heatmap(
    grid_df: pd.DataFrame,
    config_label: str,
    filename: str,
    vmin: float,
    vmax: float,
    cmap,
    norm,
):
    """Save a standalone 1×2 heatmap (SD v1.5 | FLUX) for one configuration."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), constrained_layout=True)

    sm = None
    for ax, model, show_y in zip(axes, MODELS, [True, False]):
        # Panel title = model name only; full config_label is in the suptitle
        sm = draw_panel(ax, grid_df, model, MODEL_LABELS[model], vmin, vmax, cmap, norm, show_y)

    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.85, pad=0.02)
    cbar.set_label(
        "CLCG  (EN score − Language score)\n"
        "Positive = language underperforms English   |   Negative = language outperforms English",
        fontsize=9,
    )
    cbar.ax.axhline(y=0, color="black", linewidth=1.5, linestyle="--")

    # Legend pushed below the figure — bbox_inches='tight' in savefig captures it
    na_patch = mpatches.Patch(color=NA_COLOR, label="N/A — architecturally undefined  (SD v1.5 Node 4)")
    fig.legend(handles=[na_patch], loc="lower center", ncol=1, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    fig.suptitle(
        f"MulBrandEval — Per-Node CLCG Heatmap\n{config_label}",
        fontsize=12, fontweight="bold",
    )

    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)


def save_combined_heatmap(
    grid_b: pd.DataFrame,
    grid_c: pd.DataFrame,
    vmin: float,
    vmax: float,
    cmap,
    norm,
):
    """
    Save a 2×2 combined figure:
        Row 0 (top):    Config B  |  SD v1.5   |  FLUX
        Row 1 (bottom): Config C  |  SD v1.5   |  FLUX
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    panels = [
        (grid_b, "Config B — Masked (Short-Circuit Zeros)\nSD v1.5",        "sd15", True),
        (grid_b, "Config B — Masked (Short-Circuit Zeros)\nFLUX.1-dev",     "flux", False),
        (grid_c, "Config C — Unmasked (All Nodes Measured)\nSD v1.5",       "sd15", True),
        (grid_c, "Config C — Unmasked (All Nodes Measured)\nFLUX.1-dev",    "flux", False),
    ]

    sm = None
    for ax, (grid_df, label, model, show_y) in zip(axes.flat, panels):
        sm = draw_panel(ax, grid_df, model, label, vmin, vmax, cmap, norm, show_y)

    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.7, pad=0.02)
    cbar.set_label(
        "CLCG  (EN score − Language score)\n"
        "Positive = language underperforms English   |   Negative = language outperforms English",
        fontsize=9,
    )
    cbar.ax.axhline(y=0, color="black", linewidth=1.5, linestyle="--")

    # Row labels
    for row_idx, label in enumerate(["Config B\n(Masked)", "Config C\n(Unmasked)"]):
        axes[row_idx, 0].annotate(
            label, xy=(0, 0.5), xytext=(-0.18, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=10, fontweight="bold", va="center", ha="right",
            rotation=90,
        )

    # N/A legend
    na_patch = mpatches.Patch(color=NA_COLOR, label="N/A — architecturally undefined (SD v1.5 Node 4)")
    fig.legend(handles=[na_patch], loc="lower center", fontsize=9,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)

    fig.suptitle(
        "MulBrandEval — Per-Node CLCG Heatmaps\n"
        "Configuration B (Masked, Operational DAG) vs Configuration C (Unmasked, Corrected)",
        fontsize=13, fontweight="bold",
    )

    out_path = OUTPUT_DIR / "heatmap_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)


# ── CSV output ────────────────────────────────────────────────────────────────

def save_csv(grid_b: pd.DataFrame, grid_c: pd.DataFrame):
    """
    Merge Config B and Config C grids into one 30-row precision companion CSV.
    Columns:
        node, language, model,
        en_score_b, lang_score_b, clcg_b, n_images_b,
        en_score_c, lang_score_c, clcg_c, n_images_c,
        deflation (clcg_b − clcg_c),
        node4_clcg_interpretable
    """
    b_sub = grid_b[["node", "language", "model",
                     "en_score", "lang_score", "clcg", "n_images", "interpretable"]].copy()
    b_sub = b_sub.rename(columns={
        "en_score":   "en_score_b",
        "lang_score": "lang_score_b",
        "clcg":       "clcg_b",
        "n_images":   "n_images_b",
        "interpretable": "node4_clcg_interpretable",
    })

    c_sub = grid_c[["node", "language", "model",
                     "en_score", "lang_score", "clcg", "n_images"]].copy()
    c_sub = c_sub.rename(columns={
        "en_score":   "en_score_c",
        "lang_score": "lang_score_c",
        "clcg":       "clcg_c",
        "n_images":   "n_images_c",
    })

    merged = b_sub.merge(c_sub, on=["node", "language", "model"])

    # Deflation: how much Config B overstates the gap vs Config C
    # NaN for N/A cells
    merged["deflation"] = (merged["clcg_b"] - merged["clcg_c"]).round(4)
    merged.loc[~merged["node4_clcg_interpretable"], "deflation"] = None

    # Reorder columns
    cols = [
        "node", "language", "model",
        "en_score_b", "lang_score_b", "clcg_b", "n_images_b",
        "en_score_c", "lang_score_c", "clcg_c", "n_images_c",
        "deflation", "node4_clcg_interpretable",
    ]
    merged = merged[cols]

    out_path = OUTPUT_DIR / "per_node_clcg_heatmap.csv"
    merged.to_csv(out_path, index=False)
    log.info("Saved: %s  (%d rows)", out_path, len(merged))
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # --- Validate inputs ---
    for fpath in [B_MULTILINGUAL, C_MULTILINGUAL, B_ENGLISH, C_ENGLISH]:
        if not fpath.exists():
            log.error("Missing input file: %s", fpath)
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    log.info("Loading evaluation data...")
    b_ml = pd.read_csv(B_MULTILINGUAL)
    c_ml = pd.read_csv(C_MULTILINGUAL)
    b_en = pd.read_csv(B_ENGLISH)
    c_en = pd.read_csv(C_ENGLISH)

    log.info("Config B multilingual: %d rows", len(b_ml))
    log.info("Config C multilingual: %d rows", len(c_ml))
    log.info("Config B English:      %d rows", len(b_en))
    log.info("Config C English:      %d rows", len(c_en))

    # --- Compute CLCG grids ---
    log.info("Computing CLCG grids...")
    grid_b = compute_grid(b_en, b_ml, "B")
    grid_c = compute_grid(c_en, c_ml, "C")

    # --- Shared color scale ---
    # Use all interpretable CLCG values from both configurations
    all_clcg = pd.concat([
        grid_b[grid_b["interpretable"]]["clcg"].dropna(),
        grid_c[grid_c["interpretable"]]["clcg"].dropna(),
    ])
    data_min = float(all_clcg.min())
    data_max = float(all_clcg.max())

    # Symmetric diverging scale centered at 0, with slight padding
    abs_max = max(abs(data_min), abs(data_max)) * 1.08
    vmin, vmax = -abs_max, abs_max

    log.info("Color scale: vmin=%.4f, vmax=%.4f  (data range: %.4f to %.4f)",
             vmin, vmax, data_min, data_max)

    cmap = plt.cm.RdBu_r   # Red = positive CLCG (worse), Blue = negative (better)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # --- Draw and save ---
    log.info("Generating Config B heatmap...")
    save_single_heatmap(
        grid_b,
        "Configuration B — Masked (Operational DAG with Short-Circuit Zeros)",
        "heatmap_config_b_masked.png",
        vmin, vmax, cmap, norm,
    )

    log.info("Generating Config C heatmap...")
    save_single_heatmap(
        grid_c,
        "Configuration C — Unmasked (All Nodes Measured, Corrected)",
        "heatmap_config_c_unmasked.png",
        vmin, vmax, cmap, norm,
    )

    log.info("Generating combined heatmap...")
    save_combined_heatmap(grid_b, grid_c, vmin, vmax, cmap, norm)

    # --- Save CSV ---
    log.info("Saving precision companion CSV...")
    df_csv = save_csv(grid_b, grid_c)

    # --- Quick console summary ---
    print("\n" + "=" * 70)
    print("Per-Node CLCG Summary (Config C — Unmasked, primary claim)")
    print("=" * 70)
    print(df_csv[["node", "language", "model", "clcg_c", "deflation"]].to_string(index=False))
    print()
    print("Outputs written to:", OUTPUT_DIR.resolve())
    print("  heatmap_config_b_masked.png")
    print("  heatmap_config_c_unmasked.png")
    print("  heatmap_combined.png")
    print("  per_node_clcg_heatmap.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()