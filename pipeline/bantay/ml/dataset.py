"""Labels and training samples.

The model only ever looks at *open ground*: pixels that are not dense green vegetation and not
water. That is the candidate set a monitoring run would produce ("new bare land"), so the model's
only job is the hard part: telling mine ground apart from everything else that is also bare.

- mining: open ground inside a Tang & Werner 2023 mine polygon;
- look-alikes: open ground more than 300 m from any mapped mine, labelled by its ESA WorldCover 2021
  class (cropland, built-up, bare/sparse, grass/shrub, other). "bare/sparse" includes quarries,
  riverbeds and landslide scars, which is exactly what the model must learn to reject.

Known label noise: unmapped mines (small-scale, newer than 2023) end up among the look-alikes.
That makes the model more conservative, which is the right direction for a tool that flags sites.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from ..grid import SiteGrid

NDVI_OPEN = 0.35  # below this in the dry season, ground is not dense green vegetation
NDWI_WATER = 0.1
LOOKALIKE_BUFFER_M = 300

CLASSES = ["mining", "cropland", "built", "bare_other", "grass_shrub", "other"]
# ESA WorldCover codes -> look-alike class. 80 (water) never reaches here: water is masked out.
WORLDCOVER = {10: "other", 20: "grass_shrub", 30: "grass_shrub", 40: "cropland", 50: "built",
              60: "bare_other", 90: "other", 95: "other", 100: "other"}


def mine_masks(grid: SiteGrid, shapefile: Path) -> tuple[np.ndarray, np.ndarray]:
    """(inside a mapped mine, within LOOKALIKE_BUFFER_M of one) on the grid."""
    box = gpd.GeoSeries([grid.footprint()], crs=grid.crs)
    polys = gpd.read_file(shapefile, bbox=box).to_crs(grid.crs)
    shape = (grid.height, grid.width)
    if polys.empty:
        empty = np.zeros(shape, dtype=bool)
        return empty, empty
    geoms = [g for g in polys.geometry.buffer(0) if not g.is_empty]
    inside = rasterize(geoms, out_shape=shape, transform=grid.geobox.transform, fill=0, default_value=1)
    near = rasterize([g.buffer(LOOKALIKE_BUFFER_M) for g in geoms], out_shape=shape,
                     transform=grid.geobox.transform, fill=0, default_value=1)
    return inside.astype(bool), near.astype(bool)


def open_ground(f: dict[str, np.ndarray]) -> np.ndarray:
    finite = np.all([np.isfinite(f[n]) for n in ("ndvi", "ndwi", "s2_B11")], axis=0)
    return finite & (f["land"] > 0) & (f["ndvi"] < NDVI_OPEN) & (f["ndwi"] < NDWI_WATER)


def labels(f: dict[str, np.ndarray], inside: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Class index per pixel, -1 where the pixel is not used for training."""
    y = np.full(inside.shape, -1, dtype=np.int8)
    candidate = open_ground(f)
    y[candidate & inside] = CLASSES.index("mining")
    wc = f["worldcover"].astype(np.int16)
    for code, name in WORLDCOVER.items():
        y[candidate & ~near & (wc == code)] = CLASSES.index(name)
    return y


def sample(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Flat pixel indices: up to `per_class` random pixels of each class."""
    flat = y.ravel()
    picks = []
    for k in range(len(CLASSES)):
        idx = np.flatnonzero(flat == k)
        if idx.size:
            picks.append(rng.choice(idx, size=min(per_class, idx.size), replace=False))
    return np.concatenate(picks) if picks else np.array([], dtype=np.int64)


def matrix(f: dict[str, np.ndarray], feature_names: list[str], idx: np.ndarray) -> np.ndarray:
    return np.stack([f[n].ravel()[idx] for n in feature_names], axis=1).astype(np.float32)
