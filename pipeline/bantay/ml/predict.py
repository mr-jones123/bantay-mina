"""Run the trained classifier on a region and write *unreviewed* candidate patches.

Output (in pipeline/data/ml/candidates/<region>-<year>/, not committed, not published), read by the
local review page (`npm run dev` → /review):
- candidates.geojson: every flagged patch ≥1 ha with area, calibrated probability, whether it
  touches a mine Tang & Werner already mapped, and its outline in image pixels;
- candidates.csv: the same, for spreadsheets;
- before.webp / after.webp: true-colour images of the whole region, three dry seasons apart;
- meta.json: run id, grid size and bounds.

Review decisions live in pipeline/data/ml/reviews/<region>.json, written by the review page. They
outlast re-runs: a new run re-links each earlier decision to the candidate that overlaps it most.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import transform as reproject_geom
from shapely.ops import unary_union

from ..grid import SiteGrid
from ..imagery import quality_mosaic, search
from ..render import to_rgb8
from . import HANSEN, ML_DATA, TANG_WERNER
from .dataset import mine_masks, open_ground
from .features import Region, load_or_build
from .train import MIN_PATCH_PX, p_mining

RELINK_MIN_IOU = 0.3  # an earlier decision carries over to a new patch overlapping it this much
BEFORE_YEARS = 3
STRETCH, GAMMA = (0.0, 0.22), 1.4


def _rgb(region: Region, grid: SiteGrid, year: int, f: dict | None) -> np.ndarray:
    """True-colour (H, W, 3) uint8 for a dry season; reuses the feature cache when it has the bands."""
    if f is not None:
        bands = np.stack([f["s2_B04"], f["s2_B03"], f["s2_B02"]])
    else:
        path = ML_DATA / "features" / f"{region.name}-{year}-rgb.npz"
        if path.exists():
            with np.load(path) as z:
                bands = z["rgb"]
        else:
            items = search("sentinel-2", grid, [year], region.months, region.max_scene_cloud, None)
            m, _ = quality_mosaic("sentinel-2", grid, items, ["B04", "B03", "B02"], blue="B02")
            bands = np.stack([m["B04"], m["B03"], m["B02"]])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, rgb=bands)
    return to_rgb8(bands, STRETCH, GAMMA)


def _load_reviews(region: str) -> list[dict]:
    path = ML_DATA / "reviews" / f"{region}.json"
    return json.loads(path.read_text()) if path.exists() else []


def _relink(reviews: list[dict], grid: SiteGrid, polys_utm: dict[int, object]) -> dict[int, str]:
    """candidate label -> reviewId of the earlier decision it overlaps most (IoU ≥ RELINK_MIN_IOU)."""
    to_utm = Transformer.from_crs(4326, grid.crs, always_xy=True).transform
    reviewed = [(r["reviewId"], reproject_geom(to_utm, shape(r["geometry"]))) for r in reviews]
    links: dict[int, str] = {}
    for k, poly in polys_utm.items():
        best, best_iou = None, RELINK_MIN_IOU
        for review_id, geom in reviewed:
            if not poly.intersects(geom):
                continue
            iou = poly.intersection(geom).area / poly.union(geom).area
            if iou >= best_iou:
                best, best_iou = review_id, iou
        if best:
            links[k] = best
    return links


def _rings_px(poly, grid: SiteGrid) -> list[list[list[float]]]:
    rings = []
    for part in getattr(poly, "geoms", [poly]):
        rings.append([[round(c, 1) for c in grid.to_pixel(x, y)] for x, y in part.exterior.coords])
    return rings


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
    polys_utm = {k: unary_union(parts) for k, parts in pieces.items()}
    links = _relink(_load_reviews(region.name), grid, polys_utm)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    to_ll = Transformer.from_crs(grid.crs, 4326, always_xy=True).transform
    features = []
    for k, poly_utm in polys_utm.items():
        i = k - 1
        poly = reproject_geom(to_ll, poly_utm)
        c = poly.centroid
        c0, r0 = grid.to_pixel(poly_utm.bounds[0], poly_utm.bounds[3])
        c1, r1 = grid.to_pixel(poly_utm.bounds[2], poly_utm.bounds[1])
        props = {
            "id": f"{region.name}-{year}-{k}",
            "areaHa": round(float(sizes[i] * px_ha), 2),
            "meanProbability": round(float(mean_p[i]), 3),
            "touchesMappedMine": bool(touches[i]),
            # Mean NDVI three dry seasons earlier: high means the patch was green then (recent clearing).
            "ndviThreeYearsEarlier": round(float(prev_ndvi[i]), 2),
            "lon": round(c.x, 5), "lat": round(c.y, 5),
            "pixelBBox": [round(c0, 1), round(r0, 1), round(c1, 1), round(r1, 1)],
            "ringsPx": _rings_px(poly_utm, grid),
            "reviewId": links.get(k),
        }
        features.append({"type": "Feature", "properties": props, "geometry": mapping(poly)})
    features.sort(key=lambda ft: (ft["properties"]["touchesMappedMine"],
                                  -ft["properties"]["areaHa"] * ft["properties"]["meanProbability"]))

    out = ML_DATA / "candidates" / f"{region.name}-{year}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    with open(out / "candidates.csv", "w", newline="") as fh:
        cols = [k for k in (features[0]["properties"] if features else {"id": 0}) if k not in ("ringsPx", "pixelBBox")]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for ft in features:
            w.writerow(ft["properties"])

    Image.fromarray(_rgb(region, grid, year, f)).save(out / "after.webp", quality=85)
    Image.fromarray(_rgb(region, grid, year - BEFORE_YEARS, None)).save(out / "before.webp", quality=85)
    b = grid.footprint_lonlat().bounds
    box = grid.geobox.boundingbox
    corners = [to_ll(x, y) for x, y in ((box.left, box.top), (box.right, box.top),
                                        (box.right, box.bottom), (box.left, box.bottom))]
    (out / "meta.json").write_text(json.dumps({
        "runId": run_id, "region": region.name, "year": year, "beforeYear": year - BEFORE_YEARS,
        "sizePx": [grid.width, grid.height], "pixelM": region.pixel_m,
        "bounds": [round(v, 6) for v in b], "threshold": bundle["threshold"],
        # Image corners (top-left, top-right, bottom-right, bottom-left): the grid is UTM, so the
        # image is a slightly rotated quadrilateral in lon/lat, which a map needs to place it exactly.
        "cornersLonLat": [[round(x, 6), round(y, 6)] for x, y in corners],
        "trainedOnYear": bundle["year"], "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    (out / "review.png").unlink(missing_ok=True)  # replaced by the review page

    unmapped = sum(not ft["properties"]["touchesMappedMine"] for ft in features)
    print(f"{region.name} {year}: {len(features)} flagged patches ≥1 ha, {unmapped} outside mapped mines, "
          f"{len(links)} earlier decisions re-linked -> {out}")
    return out
