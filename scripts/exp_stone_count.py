"""Experiment: deterministic CV counting of set stones vs the VLM.

Stones on gold read as LOW-saturation / HIGH-value pixels (white/rhodium)
against a saturated gold body. Touching pavé is separated with a distance
transform + watershed, the standard approach for counting touching round
objects. Deterministic and fast — the properties a VLM count lacks.

Writes an annotated image so the count can be checked by eye.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
import cv2, numpy as np

OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/stone_count.png")


def count_stones(bgr, min_area=25, max_area_frac=0.25):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Item = anything darker than the near-white studio backdrop. NOTE the
    # backdrop is bright AND desaturated, so the stone test below must be
    # applied INSIDE the item only — on the full frame it matches 86% of pixels.
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    item = (grey < 240).astype(np.uint8)
    item = cv2.morphologyEx(item, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Stone candidates: bright + desaturated (white/rhodium) inside the item,
    # against the saturated gold body.
    stone = ((s < 60) & (v > 150) & (item > 0)).astype(np.uint8) * 255
    stone = cv2.morphologyEx(stone, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Watershed split of touching stones, seeded at distance-transform peaks
    dist = cv2.distanceTransform(stone, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return 0, stone, []
    # local maxima -> markers
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    peaks = (dist == cv2.dilate(dist, kernel)) & (dist > 0.35 * dist.max())
    n_mark, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    markers = markers + 1
    markers[stone == 0] = 1                      # background
    cv2.watershed(bgr, markers)

    total = stone.sum() / 255.0
    boxes = []
    for lab in range(2, markers.max() + 1):
        m = (markers == lab)
        area = int(m.sum())
        if area < min_area or area > total * max_area_frac + 1:
            continue
        ys, xs = np.where(m)
        boxes.append((int(xs.min()), int(ys.min()),
                      int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), area))
    return len(boxes), stone, boxes


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/cases/8ce4f55e/img_0.jpg"
    bgr = cv2.imread(str(ROOT / path))
    if bgr is None:
        print("cannot read", path); return
    # Stored case photos carry a dark KANCHAN caption band at the bottom —
    # drop it so it isn't segmented as part of the item.
    grey_rows = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean(axis=1)
    dark = np.where(grey_rows < 140)[0]
    if len(dark) and dark.min() > bgr.shape[0] * 0.6:
        bgr = bgr[:dark.min(), :]
    n, mask, boxes = count_stones(bgr)
    print(f"image {path}  -> {n} stones detected (deterministic)")
    areas = sorted(b[4] for b in boxes)
    print("areas px:", areas)

    vis = bgr.copy()
    for i, (x, y, w, hh, a) in enumerate(sorted(boxes, key=lambda b: (b[1], b[0])), 1):
        cv2.rectangle(vis, (x, y), (x + w, y + hh), (0, 0, 255), 1)
        cv2.putText(vis, str(i), (x, max(0, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT), vis)
    print("annotated ->", OUT)


if __name__ == "__main__":
    main()
