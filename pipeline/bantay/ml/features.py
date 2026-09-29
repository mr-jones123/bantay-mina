"""Per-pixel feature stack for one region and one dry season.

Everything is resampled onto one UTM grid (20 m by default) and cached as a compressed .npz, so
training and prediction read identical inputs. Features are the traits mining land shares beyond
"trees are gone":

- what the ground is made of: iron oxide (red/blue), clays (SWIR1/SWIR2), bare-soil index;
- whether it stays bare: this season vs. three seasons earlier, and years since Hansen tree loss;
- its texture and surroundings: local variability and how much bare ground surrounds it;
- radar backscatter (Sentinel-1), which sees benches, waste rock and plant through cloud;
- terrain: mines cut into slopes, rice fields sit on flat floodplains.
"""

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from odc.stac import load
from scipy import ndimage
from scipy.ndimage import uniform_filter

from .. import forest
from ..grid import SiteGrid, site_grid
from ..imagery import _catalog, quality_mosaic, search

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
FEATURE_VERSION = 1


@dataclass(frozen=True)
class Region:
    name: str
    center: tuple[float, float]
    box_km: float
    pixel_m: float
    months: tuple[int, int]
    max_scene_cloud: float

    def grid(self) -> SiteGrid:
        return site_grid(self.center, self.box_km, self.pixel_m)


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a / b).astype(np.float32)


def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((a - b) / (a + b)).astype(np.float32)


def _local_std(x: np.ndarray, size: int) -> np.ndarray:
    filled = np.nan_to_num(x, nan=np.nanmean(x))
    mean = uniform_filter(filled, size)
    sq = uniform_filter(filled * filled, size)
    return np.sqrt(np.clip(sq - mean * mean, 0, None)).astype(np.float32)


def _s2(region: Region, grid: SiteGrid, year: int) -> dict[str, np.ndarray]:
    items = search("sentinel-2", grid, [year], region.months, region.max_scene_cloud, None)
    m, counts = quality_mosaic("sentinel-2", grid, items, S2_BANDS, blue="B02")
    m["clear_views"] = counts.astype(np.float32)
    return m


def _s1(grid: SiteGrid, year: int, months: tuple[int, int]) -> dict[str, np.ndarray]:
    end = date(year + (months[1] == 12), months[1] % 12 + 1, 1)
    items = _catalog().search(
        collections=["sentinel-1-rtc"],
        intersects=grid.footprint_lonlat().__geo_interface__,
        datetime=f"{date(year, months[0], 1).isoformat()}/{end.isoformat()}",
    ).item_collection()
    items = [i for i in items if {"vv", "vh"} <= set(i.assets)]  # skip single-polarisation scenes
    if not items:
        raise RuntimeError(f"no dual-polarisation Sentinel-1 RTC scenes for {year}")
    ds = load(items, bands=["vv", "vh"], geobox=grid.geobox, groupby="solar_day",
              resampling="bilinear", fail_on_error=False, pool=16)
    out = {}
    for pol in ("vv", "vh"):
        power = ds[pol].values.astype(np.float32)
        power[~np.isfinite(power) | (power <= 0)] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            db = 10 * np.log10(power)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels stay NaN
            out[f"s1_{pol}_db"] = np.nanmedian(db, axis=0).astype(np.float32)
            out[f"s1_{pol}_std"] = np.nanstd(db, axis=0).astype(np.float32)
    out["s1_vh_minus_vv"] = out["s1_vh_db"] - out["s1_vv_db"]
    return out


def _dem(grid: SiteGrid) -> dict[str, np.ndarray]:
    items = _catalog().search(collections=["cop-dem-glo-30"],
                              intersects=grid.footprint_lonlat().__geo_interface__).item_collection()
    ds = load(items, bands=["data"], geobox=grid.geobox, resampling="bilinear", pool=8)
    elev = ds["data"].values.max(axis=0).astype(np.float32)  # tiles do not overlap
    dy, dx = np.gradient(elev, grid.pixel_m)
    slope = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    return {"elevation": elev, "slope": slope}


def worldcover(grid: SiteGrid) -> np.ndarray:
    """ESA WorldCover 2021 class codes on the grid (used only for labels, never as a feature)."""
    items = _catalog().search(collections=["esa-worldcover"],
                              intersects=grid.footprint_lonlat().__geo_interface__,
                              datetime="2021-01-01/2021-12-31").item_collection()
    ds = load(items, bands=["map"], geobox=grid.geobox, resampling="mode", pool=8)
    return ds["map"].values.max(axis=0).astype(np.uint8)


def build(region: Region, year: int, hansen_cache: Path) -> dict[str, np.ndarray]:
    grid = region.grid()
    now = _s2(region, grid, year)
    prev = _s2(region, grid, year - 3)
    f: dict[str, np.ndarray] = {f"s2_{b}": now[b] for b in S2_BANDS}
    b02, b03, b04, b08, b11, b12 = (now[b] for b in S2_BANDS)
    f["ndvi"] = _norm_diff(b08, b04)
    f["ndwi"] = _norm_diff(b03, b08)
    f["bsi"] = _norm_diff(b11 + b04, b08 + b02)
    f["iron_oxide"] = _ratio(b04, b02)  # laterite and oxidised waste are red
    f["ferrous"] = _ratio(b11, b08)
    f["clay"] = _ratio(b11, b12)  # hydrothermal clays absorb in SWIR2
    f["brightness"] = ((b02 + b03 + b04) / 3).astype(np.float32)
    f["clear_views"] = now["clear_views"]

    f["ndvi_prev"] = _norm_diff(prev["B08"], prev["B04"])
    f["bsi_prev"] = _norm_diff(prev["B11"] + prev["B04"], prev["B08"] + prev["B02"])
    f["d_ndvi"] = f["ndvi"] - f["ndvi_prev"]

    bare = (f["ndvi"] < 0.3).astype(np.float32)
    f["bare_frac_220m"] = uniform_filter(bare, 11).astype(np.float32)
    f["bare_frac_500m"] = uniform_filter(bare, 25).astype(np.float32)
    f["b12_std_100m"] = _local_std(b12, 5)
    f["ndvi_std_100m"] = _local_std(f["ndvi"], 5)

    f.update(_s1(grid, year, region.months))
    f.update(_dem(grid))

    loss = forest.layer_on_grid("lossyear", grid, hansen_cache).astype(np.float32)
    f["years_since_tree_loss"] = np.where(loss > 0, year - (2000 + loss), -1).astype(np.float32)
    f["treecover2000"] = forest.layer_on_grid("treecover2000", grid, hansen_cache).astype(np.float32)
    f["land"] = forest.land_mask(grid, hansen_cache).astype(np.float32)
    return f


def cache_path(root: Path, region: str, year: int) -> Path:
    return root / "features" / f"{region}-{year}-v{FEATURE_VERSION}.npz"


def _derived(f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Cheap features computed from the cache on load, so changing them needs no re-download."""
    # Terrain position: height relative to the surrounding 500 m. Pits sit low, waste dumps high.
    # Absolute elevation is left out on purpose: the first model leaned on it as a shortcut (the
    # training mines happen to be low; Benguet's are high), which does not generalise.
    f["tpi_500m"] = (f["elevation"] - uniform_filter(f["elevation"], 25)).astype(np.float32)
    return f


def _surface_water(grid: SiteGrid) -> dict[str, np.ndarray]:
    """JRC Global Surface Water occurrence 1984–2020: % of months each pixel was water.

    Riverbeds, sandbars and beaches are bare in the dry season but wet in some months; mine pits
    and dumps mostly are not. The first model confused riverbeds and beaches with mine ground.
    """
    items = _catalog().search(collections=["jrc-gsw"],
                              intersects=grid.footprint_lonlat().__geo_interface__).item_collection()
    ds = load(items, bands=["occurrence"], geobox=grid.geobox, resampling="bilinear", pool=8)
    occ = ds["occurrence"].values.max(axis=0).astype(np.float32)
    occ[occ > 100] = 0  # 255 = no data (never water)
    return {"water_occurrence": occ,
            "water_occurrence_max_100m": ndimage.maximum_filter(occ, size=5).astype(np.float32)}


def _static(region: Region, root: Path) -> dict[str, np.ndarray]:
    """Year-independent layers, cached once per region."""
    path = root / "features" / f"{region.name}-static-v{FEATURE_VERSION}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    s = _surface_water(region.grid())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **s)
    return s


def load_or_build(region: Region, year: int, root: Path, hansen_cache: Path) -> dict[str, np.ndarray]:
    path = cache_path(root, region.name, year)
    if path.exists():
        with np.load(path) as z:
            f = {k: z[k] for k in z.files}
    else:
        f = build(region, year, hansen_cache)
        f["worldcover"] = worldcover(region.grid()).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **f)
    f.update(_static(region, root))
    return _derived(f)


# Columns the model sees. Everything else in the cache (land, worldcover) is for masking/labels.
FEATURES = [
    *(f"s2_{b}" for b in S2_BANDS),
    "ndvi", "ndwi", "bsi", "iron_oxide", "ferrous", "clay", "brightness",
    "ndvi_prev", "bsi_prev", "d_ndvi",
    "bare_frac_220m", "bare_frac_500m", "b12_std_100m", "ndvi_std_100m",
    "s1_vv_db", "s1_vh_db", "s1_vv_std", "s1_vh_std", "s1_vh_minus_vv",
    "slope", "tpi_500m",
    "water_occurrence", "water_occurrence_max_100m",
    "years_since_tree_loss", "treecover2000",
]
