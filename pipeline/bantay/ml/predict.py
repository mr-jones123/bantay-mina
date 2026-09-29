"""Run the trained classifier on a region and write *unreviewed* candidate patches.

Output (in pipeline/data/ml/candidates/<region>-<year>/, not committed, not published):
- candidates.geojson: every flagged patch ≥1 ha, with area, calibrated probability, and whether it
  touches a mine that Tang & Werner already mapped;
- candidates.csv: the same, sorted for review;
- review.png: true-colour thumbnails of the patches outside mapped mines, for a person to check.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import transform as reproject_geom
from shapely.ops import unary_union

from ..render import to_rgb8
from . import HANSEN, ML_DATA, TANG_WERNER
from .dataset import mine_masks, open_ground
from .features import Region, load_or_build
from .train import MIN_PATCH_PX, p_mining

THUMB_PX = 160
MAX_THUMBS = 30


def run(region: Region, year: int) -> Path:
    bundle = joblib.load(ML_DATA / "model" / "model.joblib")
    grid = region.grid()
    f = load_or_build(region, year, ML_DATA, HANSEN)
    _, near = mine_masks(grid, TANG_WERNER)
    cand = open_ground(f)
    raw = p_mining(bundle["model"], "lightgbm", f, cand)
    prob = np.zeros(cand.shape, dtype=np.float32)
    prob.ravel()[np.flatnonzero(cand.ravel())] = bundle["calibration"].predict(raw)
    flagged = np.zeros(cand.shape, dtype=bool)
    flagged.ravel()[np.flatnonzero(cand.ravel())] = raw >= bundle["threshold"]

    lab, n = ndimage.label(flagged, structure=np.ones((3, 3)))
    ids = np.arange(1, n + 1)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=ids)
    mean_p = ndimage.mean(prob, lab, index=ids)
    touches = ndimage.maximum(near, lab, index=ids)
    prev_ndvi = ndimage.mean(np.nan_to_num(f["ndvi_prev"], nan=0), lab, index=ids)
    px_ha = region.pixel_m ** 2 / 10_000
    keep = {int(i) for i, s in zip(ids, sizes) if s >= MIN_PATCH_PX}

    # One (multi)polygon per patch: `shapes` returns one piece per 8-connected run, so gather them.
    pieces: dict[int, list] = {}
    for geom, value in shapes(lab.astype(np.int32), mask=np.isin(lab, list(keep)),
                              connectivity=8, transform=grid.geobox.transform):
        pieces.setdefault(int(value), []).append(shape(geom))

    to_ll = Transformer.from_crs(grid.crs, 4326, always_xy=True).transform
    features, rows = [], []
    for k, parts in pieces.items():
        i = k - 1
        poly_utm = unary_union(parts)
        poly = reproject_geom(to_ll, poly_utm)
        c = poly.centroid
        props = {
            "id": f"{region.name}-{k}",
            "areaHa": round(float(sizes[i] * px_ha), 2),
            "meanProbability": round(float(mean_p[i]), 3),
            "touchesMappedMine": bool(touches[i]),
            # Mean NDVI three dry seasons earlier: high means the patch was green then (recent clearing).
            "ndviThreeYearsEarlier": round(float(prev_ndvi[i]), 2),
            "lon": round(c.x, 5), "lat": round(c.y, 5),
            "review": "unreviewed",
        }
        features.append({"type": "Feature", "properties": props, "geometry": mapping(poly)})
        rows.append((props, poly_utm))

    out = ML_DATA / "candidates" / f"{region.name}-{year}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    rows.sort(key=lambda r: (r[0]["touchesMappedMine"], -r[0]["areaHa"] * r[0]["meanProbability"]))
    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0][0]) if rows else ["id"])
        w.writeheader()
        for props, _ in rows:
            w.writerow(props)
    _review_sheet(f, grid, [r for r in rows if not r[0]["touchesMappedMine"]][:MAX_THUMBS], out / "review.png")
    unmapped = sum(not r[0]["touchesMappedMine"] for r in rows)
    print(f"{region.name} {year}: {len(rows)} flagged patches ≥1 ha, {unmapped} outside mapped mines -> {out}")
    return out


def _review_sheet(f, grid, rows, path: Path) -> None:
    if not rows:
        return
    rgb8 = to_rgb8(np.stack([f["s2_B04"], f["s2_B03"], f["s2_B02"]]), (0.0, 0.22), 1.4)
    font = ImageFont.load_default(size=11)
    cols = 6
    sheet = Image.new("RGB", (cols * THUMB_PX, ((len(rows) + cols - 1) // cols) * (THUMB_PX + 28)), "white")
    for n, (props, poly) in enumerate(rows):
        minx, miny, maxx, maxy = poly.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        half = max(maxx - minx, maxy - miny) / 2 + 300  # 300 m of context around the patch
        c0, r0 = grid.to_pixel(cx - half, cy + half)
        c1, r1 = grid.to_pixel(cx + half, cy - half)
        c0, r0 = max(0, int(c0)), max(0, int(r0))
        c1, r1 = min(grid.width, int(c1)), min(grid.height, int(r1))
        if c1 - c0 < 2 or r1 - r0 < 2:
            continue
        crop = Image.fromarray(rgb8[r0:r1, c0:c1]).resize((THUMB_PX, THUMB_PX), Image.NEAREST)
        draw = ImageDraw.Draw(crop)
        sx, sy = THUMB_PX / max(1, c1 - c0), THUMB_PX / max(1, r1 - r0)
        for part in getattr(poly, "geoms", [poly]):
            ring = [((grid.to_pixel(x, y)[0] - c0) * sx, (grid.to_pixel(x, y)[1] - r0) * sy)
                    for x, y in part.exterior.coords]
            draw.line(ring, fill=(0, 255, 255), width=2)
        x, y = (n % cols) * THUMB_PX, (n // cols) * (THUMB_PX + 28)
        sheet.paste(crop, (x, y))
        d = ImageDraw.Draw(sheet)
        d.text((x + 3, y + THUMB_PX + 2), props["id"], font=font, fill="black")
        d.text((x + 3, y + THUMB_PX + 14),
               f"{props['areaHa']} ha  p={props['meanProbability']}  NDVI-3y {props['ndviThreeYearsEarlier']}",
               font=font, fill="black")
    sheet.save(path)
