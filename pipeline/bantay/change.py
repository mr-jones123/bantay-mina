"""Change measures between the two composites of one comparison.

Vegetation -> bare is independent of the Hansen dataset: Hansen counts only stand-replacing
loss of trees taller than 5 m, so it misses clearing of farms, brush and young regrowth that
mining also removes. This measure counts any pixel that was clearly green before and is
clearly not green after, whatever grew there.

Water change catches what vegetation measures cannot: new ponds (tailings, pits), and
shorelines or river mouths that moved, as at black sand mining sites.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.features import geometry_mask
from shapely.geometry.base import BaseGeometry

from .grid import SiteGrid

NDVI_VEGETATED = 0.5  # dense green vegetation in the dry-season composite
NDVI_BARE = 0.3  # bare soil, rock, water, pavement, or very sparse cover
COLOUR = (255, 64, 214)  # magenta, distinct from the yellow–red Hansen ramp


def vegetation_to_bare(before_ndvi: np.ndarray, after_ndvi: np.ndarray, land: np.ndarray,
                       grid: SiteGrid, footprint_utm: BaseGeometry | None, overlay: Path) -> dict:
    # Open water gives noisy NDVI; permanent water (Hansen datamask) is left out on both sides.
    valid = land & np.isfinite(before_ndvi) & np.isfinite(after_ndvi)
    vegetated = valid & (before_ndvi >= NDVI_VEGETATED)
    changed = vegetated & (after_ndvi < NDVI_BARE)
    px_ha = grid.pixel_m**2 / 10_000

    stats = {
        "ndviVegetated": NDVI_VEGETATED,
        "ndviBare": NDVI_BARE,
        "comparableFraction": round(float(valid.mean()), 4),
        "vegetatedBeforeHa": round(float(vegetated.sum() * px_ha), 1),
        "vegetationToBareHa": round(float(changed.sum() * px_ha), 1),
        "insideFootprintHa": None,
        "overlay": None,
    }
    if footprint_utm is not None:
        inside = geometry_mask([footprint_utm], out_shape=changed.shape,
                               transform=grid.geobox.transform, invert=True)
        stats["insideFootprintHa"] = round(float((changed & inside).sum() * px_ha), 1)

    rgba = np.zeros((*changed.shape, 4), dtype=np.uint8)
    rgba[changed] = (*COLOUR, 190)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(overlay, optimize=True)
    return stats


# McFeeters NDWI = (green - NIR) / (green + NIR). A gap between the two thresholds keeps pixels
# that hover around zero (wet soil, shadow, turbid shallows) from flickering into "change".
NDWI_WATER = 0.1
NDWI_DRY = -0.1
# Open sea is so dark that NDWI becomes noise (both bands near zero, sometimes below zero after
# the reflectance offset), so inside mapped permanent water, darkness alone means water. On land,
# darkness means deep terrain shadow instead, so there only NDWI decides. "Dry" must reflect a little.
NIR_DARK_WATER = 0.03
NIR_DRY_MIN = 0.05
TO_WATER_COLOUR = (47, 123, 255)  # blue
TO_LAND_COLOUR = (157, 255, 58)  # lime


def _water(ndwi: np.ndarray, nir: np.ndarray, land: np.ndarray) -> np.ndarray:
    return (ndwi > NDWI_WATER) | (~land & (nir < NIR_DARK_WATER))


def _dry(ndwi: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (ndwi < NDWI_DRY) & (nir >= NIR_DRY_MIN)


def water_change(before_ndwi: np.ndarray, before_nir: np.ndarray, after_ndwi: np.ndarray,
                 after_nir: np.ndarray, land: np.ndarray, grid: SiteGrid,
                 footprint_utm: BaseGeometry | None, overlay: Path) -> dict:
    valid = np.isfinite(before_ndwi) & np.isfinite(after_ndwi)
    # Only ground that was land in the 2000s (Hansen datamask) can newly become water. Old Landsat
    # surface reflectance over open sea is unreliable enough to make the sea look "dry" otherwise.
    to_water = valid & land & _dry(before_ndwi, before_nir) & _water(after_ndwi, after_nir, land)
    to_land = valid & _water(before_ndwi, before_nir, land) & _dry(after_ndwi, after_nir)
    px_ha = grid.pixel_m**2 / 10_000

    stats = {
        "ndwiWater": NDWI_WATER,
        "ndwiDry": NDWI_DRY,
        "landToWaterHa": round(float(to_water.sum() * px_ha), 1),
        "waterToLandHa": round(float(to_land.sum() * px_ha), 1),
        "landToWaterInsideFootprintHa": None,
        "overlay": None,
    }
    if footprint_utm is not None:
        inside = geometry_mask([footprint_utm], out_shape=to_water.shape,
                               transform=grid.geobox.transform, invert=True)
        stats["landToWaterInsideFootprintHa"] = round(float((to_water & inside).sum() * px_ha), 1)

    rgba = np.zeros((*to_water.shape, 4), dtype=np.uint8)
    rgba[to_water] = (*TO_WATER_COLOUR, 200)
    rgba[to_land] = (*TO_LAND_COLOUR, 200)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(overlay, optimize=True)
    return stats
