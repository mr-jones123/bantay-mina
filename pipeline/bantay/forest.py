"""Tree cover loss from Hansen et al. Global Forest Change (University of Maryland)."""

import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from shapely.geometry.base import BaseGeometry

from .grid import SiteGrid

VERSION = "GFC-2025-v1.13"
FIRST_YEAR, LAST_YEAR = 2001, 2025
CANOPY_THRESHOLD = 30  # % canopy in 2000 counted as "tree cover" (Global Forest Watch default)
BASE_URL = f"https://storage.googleapis.com/earthenginepartners-hansen/{VERSION}"
CITATION = (
    "Hansen, M. C. et al. 2013. High-Resolution Global Maps of 21st-Century Forest Cover Change. "
    f"Science 342: 850–53. Data: {VERSION}, University of Maryland (CC BY 4.0)."
)
EARTH_RADIUS_M = 6_371_007.2


def _tiles(bounds: tuple[float, float, float, float]) -> list[str]:
    """Names of the 10° Hansen tiles (named by their top-left corner) covering `bounds`."""
    west, south, east, north = bounds
    names = []
    for top in range(math.ceil(south / 10) * 10, math.ceil(north / 10) * 10 + 1, 10):
        for left in range(math.floor(west / 10) * 10, math.floor(east / 10) * 10 + 1, 10):
            ns = f"{abs(top):02d}{'N' if top >= 0 else 'S'}"
            ew = f"{abs(left):03d}{'E' if left >= 0 else 'W'}"
            names.append(f"{ns}_{ew}")
    return names


def _fetch(layer: str, tile: str, cache: Path) -> Path:
    name = f"Hansen_{VERSION}_{layer}_{tile}.tif"
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {name} (one-off, cached in {cache})")
        tmp = path.with_suffix(f".{os.getpid()}.part")  # parallel builds may fetch the same tile
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", tmp)
        tmp.rename(path)
    return path


def _read(layer: str, bounds: tuple[float, float, float, float], cache: Path):
    """Mosaic of one Hansen layer over `bounds`, fetching every tile it touches."""
    sources = [rasterio.open(_fetch(layer, tile, cache)) for tile in _tiles(bounds)]
    try:
        data, transform = merge(sources, bounds=bounds)
        return data[0], transform, sources[0].crs
    finally:
        for src in sources:
            src.close()


def _pixel_area_ha(transform, shape: tuple[int, int]) -> np.ndarray:
    """Area of each lat/lon pixel row (spherical approximation), broadcast to the window."""
    rows = np.arange(shape[0]) + 0.5
    lat = np.radians(transform.f + transform.e * rows)
    dy = EARTH_RADIUS_M * math.radians(abs(transform.e))
    dx = EARTH_RADIUS_M * math.radians(transform.a) * np.cos(lat)
    return np.broadcast_to((dx * dy / 10_000)[:, None], shape)


def _stats(inside: np.ndarray, cover: np.ndarray, loss: np.ndarray, area: np.ndarray) -> dict:
    forest = inside & (cover >= CANOPY_THRESHOLD)
    by_year = {
        str(FIRST_YEAR + y - 1): round(float(area[forest & (loss == y)].sum()), 1)
        for y in range(1, LAST_YEAR - FIRST_YEAR + 2)
    }
    return {
        "areaHa": round(float(area[inside].sum()), 1),
        "treeCover2000Ha": round(float(area[forest].sum()), 1),
        "lossHa": round(sum(by_year.values()), 1),
        "lossByYear": by_year,
    }


def forest_loss(grid: SiteGrid, footprint_lonlat: BaseGeometry | None, cache: Path):
    """Returns (stats, loss-year raster on the site grid for the overlay)."""
    box_ll = grid.footprint_lonlat()
    bounds = box_ll.bounds
    cover, transform, crs = _read("treecover2000", bounds, cache)
    loss, _, _ = _read("lossyear", bounds, cache)
    area = _pixel_area_ha(transform, cover.shape)

    def mask_of(geom: BaseGeometry) -> np.ndarray:
        return geometry_mask([geom], out_shape=cover.shape, transform=transform, invert=True)

    stats = {
        "dataset": f"Hansen {VERSION}",
        "citation": CITATION,
        "canopyThresholdPct": CANOPY_THRESHOLD,
        "years": [FIRST_YEAR, LAST_YEAR],
        "box": _stats(mask_of(box_ll), cover, loss, area),
        "footprint": _stats(mask_of(footprint_lonlat), cover, loss, area) if footprint_lonlat else None,
    }

    # Overlay: loss pixels (on >=30% canopy) reprojected onto the image grid.
    counted = np.where(cover >= CANOPY_THRESHOLD, loss, 0).astype(np.uint8)
    on_grid = np.zeros((grid.height, grid.width), dtype=np.uint8)
    reproject(counted, on_grid, src_transform=transform, src_crs=crs,
              dst_transform=grid.geobox.transform, dst_crs=grid.crs.to_wkt(),
              resampling=Resampling.nearest)
    on_grid = np.where(on_grid > 0, on_grid.astype(np.int16) + FIRST_YEAR - 1, 0)
    return stats, on_grid


def layer_on_grid(layer: str, grid: SiteGrid, cache: Path) -> np.ndarray:
    """One Hansen layer (e.g. lossyear, treecover2000, datamask) resampled onto the site grid."""
    data, transform, crs = _read(layer, grid.footprint_lonlat().bounds, cache)
    on_grid = np.zeros((grid.height, grid.width), dtype=data.dtype)
    reproject(data, on_grid, src_transform=transform, src_crs=crs,
              dst_transform=grid.geobox.transform, dst_crs=grid.crs.to_wkt(),
              resampling=Resampling.nearest)
    return on_grid


def land_mask(grid: SiteGrid, cache: Path) -> np.ndarray:
    """True where Hansen's datamask marks land (1), i.e. not permanent water (2) or no data (0).

    The datamask reflects the 2000s, so ponds dug since then (tailings, pits) still count as land.
    """
    return layer_on_grid("datamask", grid, cache) == 1
