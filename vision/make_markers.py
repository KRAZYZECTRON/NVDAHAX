"""Generate printable ArUco markers (DICT_4X4_50) and a calibration checkerboard.

Writes vision/markers/markers.pdf (one marker per A4 page) and
vision/markers/checkerboard.pdf. Print at 100% / "Actual size", NOT "fit to page",
then check the 10 cm ruler line on each page with a real ruler.

Usage:  python -m vision.make_markers [--rover-cm 9] [--corner-cm 15]
"""
import argparse
import os

import cv2
import numpy as np

DPI = 300
A4_PX = (2480, 3508)  # portrait (w, h) at 300 dpi
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "markers")


def cm_to_px(cm):
    return int(round(cm / 2.54 * DPI))


def text(img, s, y, scale=2.0, thick=4):
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, s, ((img.shape[1] - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, 0, thick, cv2.LINE_AA)


def ruler(img, y):
    x0 = (img.shape[1] - cm_to_px(10)) // 2
    x1 = x0 + cm_to_px(10)
    cv2.line(img, (x0, y), (x1, y), 0, 4)
    for x in (x0, x1):
        cv2.line(img, (x, y - 30), (x, y + 30), 0, 4)
    text(img, "this line must measure exactly 10.0 cm", y + 90, 1.4, 3)


def marker_page(dictionary, marker_id, size_cm, is_rover):
    page = np.full((A4_PX[1], A4_PX[0]), 255, np.uint8)
    side = cm_to_px(size_cm)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side, borderBits=1)
    x0 = (A4_PX[0] - side) // 2
    y0 = 600
    page[y0:y0 + side, x0:x0 + side] = marker

    # Thin grey cut line 1 cm outside the marker (keep this white quiet zone).
    m = cm_to_px(1.0)
    cv2.rectangle(page, (x0 - m, y0 - m), (x0 + side + m, y0 + side + m), 170, 2)

    if is_rover:
        tip = (A4_PX[0] // 2, y0 - m - 40)
        cv2.arrowedLine(page, (tip[0], tip[1] - 0), (tip[0], tip[1] - 250), 0, 8, tipLength=0.3)
        text(page, "FRONT OF ROVER", y0 - m - 320, 2.0, 5)
        label = f"ROVER  ID {marker_id}  ({size_cm:.1f} cm)"
    else:
        label = f"CORNER  ID {marker_id}  ({size_cm:.1f} cm)"
    text(page, label, y0 + side + m + 150, 2.5, 6)
    text(page, "DICT_4X4_50 - print at 100% (Actual size)", y0 + side + m + 260, 1.4, 3)
    ruler(page, A4_PX[1] - 400)
    return page


def checkerboard_page(cols, rows, square_mm):
    """cols x rows SQUARES -> (cols-1) x (rows-1) inner corners. A4 landscape."""
    page = np.full((A4_PX[0], A4_PX[1]), 255, np.uint8)
    sq = int(round(square_mm / 25.4 * DPI))
    bw, bh = cols * sq, rows * sq
    x0, y0 = (page.shape[1] - bw) // 2, (page.shape[0] - bh) // 2 - 100
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                page[y0 + r * sq:y0 + (r + 1) * sq, x0 + c * sq:x0 + (c + 1) * sq] = 0
    text(page, f"{cols - 1}x{rows - 1} inner corners, {square_mm} mm squares - print at 100%",
         y0 + bh + 110, 1.4, 3)
    ruler(page, page.shape[0] - 150)
    return page


def save_pdf(pages, path):
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; skipping PDF (PNGs still written). pip install pillow")
        return
    imgs = [Image.fromarray(p).convert("L") for p in pages]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], resolution=DPI)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rover-cm", type=float, default=9.0, help="rover marker side (8-10 cm)")
    ap.add_argument("--corner-cm", type=float, default=15.0, help="corner marker side")
    ap.add_argument("--square-mm", type=float, default=25.0, help="checkerboard square size")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    pages = [marker_page(dictionary, 0, args.rover_cm, True)]
    pages += [marker_page(dictionary, i, args.corner_cm, False) for i in (1, 2, 3, 4)]
    for i, p in enumerate(pages):
        cv2.imwrite(os.path.join(OUT_DIR, f"marker_{i}.png"), p)
    save_pdf(pages, os.path.join(OUT_DIR, "markers.pdf"))

    board = checkerboard_page(10, 7, args.square_mm)
    cv2.imwrite(os.path.join(OUT_DIR, "checkerboard.png"), board)
    save_pdf([board], os.path.join(OUT_DIR, "checkerboard.pdf"))
    print("PNGs in", OUT_DIR)


if __name__ == "__main__":
    main()
