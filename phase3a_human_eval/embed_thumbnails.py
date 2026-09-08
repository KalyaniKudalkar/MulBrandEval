"""
MulBrandEval — Phase 3A: Embed image thumbnails into the blind rating tracker
================================================================================
Reads Phase3A_Primary_Rater_Tracker_Blind.xlsx, finds each row's Rating ID
(e.g. R1-001), locates the matching image in the staged folders, and inserts
an actual thumbnail into column C of that row. Saves the result as a NEW file
so the original template is untouched.

Run from project root, AFTER running phase3a_stage_120_sample_blind.py:
    pip install pillow  (if not already installed)
    python embed_thumbnails.py
"""

from pathlib import Path
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from PIL import Image as PILImage

TRACKER_IN = "phase3a_human_eval/Phase3A_Primary_Rater_Tracker.xlsx"
TRACKER_OUT = "phase3a_human_eval/Phase3A_Primary_Rater_Tracker_WITH_IMAGES.xlsx"

STAGING_DIR_120 = Path("data/phase3a_primary_rater_staging_pass1")
STAGING_DIR_20 = Path("data/phase3a_primary_rater_staging_pass2")

THUMB_SIZE = (240, 240)  # pixels, fits the 190pt row height set in the v2 template


def make_thumbnail(src_path, tmp_path):
    """Resize a copy of the image so the embedded version stays small/fast."""
    img = PILImage.open(src_path)
    img.thumbnail(THUMB_SIZE)
    img.save(tmp_path)


def embed_sheet(ws, staging_dir, tmp_dir):
    count = 0
    for row in range(2, ws.max_row + 1):
        rating_id = ws.cell(row=row, column=2).value
        if not rating_id:
            continue
        src = staging_dir / f"{rating_id}.png"
        if not src.exists():
            print(f"MISSING image for {rating_id}: {src}")
            continue
        tmp_thumb = tmp_dir / f"{rating_id}_thumb.png"
        make_thumbnail(src, tmp_thumb)
        img = XLImage(str(tmp_thumb))
        img.width, img.height = THUMB_SIZE
        anchor = f"C{row}"
        ws.add_image(img, anchor)
        count += 1
    print(f"Embedded {count} thumbnails into '{ws.title}'")


def main():
    tmp_dir = Path("data/.thumb_cache")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    wb = load_workbook(TRACKER_IN)

    ws1 = wb["Pass 1 - 120 Images"]
    embed_sheet(ws1, STAGING_DIR_120, tmp_dir)

    ws2 = wb["Pass 2 - Intrarater 20"]
    embed_sheet(ws2, STAGING_DIR_20, tmp_dir)

    wb.save(TRACKER_OUT)
    print(f"\nSaved: {TRACKER_OUT}")
    print("Open this file to rate. The original template file is untouched.")


if __name__ == "__main__":
    main()