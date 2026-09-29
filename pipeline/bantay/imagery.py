"""Cloud-masked, dry-season composites from Microsoft Planetary Computer."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import planetary_computer
import pystac_client
from odc.stac import load

from .grid import SiteGrid

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Landsat Collection 2 Level-2 QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 shadow.
LANDSAT_BAD_BITS = 0b11111
# Sentinel-2 scene classification classes kept: dark/terrain, vegetation, bare, water, unclassified.
S2_CLEAR_CLASSES = [2, 4, 5, 6, 7]

SENSORS = {
    "landsat": {
        "collection": "landsat-c2-l2",
        "bands": ["red", "green", "blue", "nir08"],
        "mask_band": "qa_pixel",
        "native_m": 30,
        "attribution": "Landsat imagery courtesy of USGS/NASA",
    },
    "sentinel-2": {
        "collection": "sentinel-2-l2a",
        "bands": ["B04", "B03", "B02", "B08"],
        "mask_band": "SCL",
        "native_m": 10,
        "attribution": "Contains modified Copernicus Sentinel data",
    },
}

PLATFORM_NAMES = {
    "landsat-5": "Landsat 5",
    "landsat-7": "Landsat 7",
    "landsat-8": "Landsat 8",
    "landsat-9": "Landsat 9",
}


@dataclass
class Composite:
    rgb: np.ndarray  # float32 (3, H, W) surface reflectance, NaN where never clear
    ndvi: np.ndarray  # float32 (H, W) NDVI of the same chosen view, NaN where never clear
    ndwi: np.ndarray  # float32 (H, W) McFeeters NDWI (green vs NIR) of the same chosen view
    nir: np.ndarray  # float32 (H, W) near-infrared reflectance of the same chosen view
    sensor: str
    platforms: list[str]
    scene_ids: list[str]
    scene_dates: list[str]
    clear_fraction: float  # share of pixels with at least one clear observation
    median_clear_obs: float


def _catalog() -> pystac_client.Client:
    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    client.add_conforms_to("FILTER")  # Planetary Computer supports CQL2 but does not advertise it
    return client


def search(sensor: str, grid: SiteGrid, years: list[int], months: tuple[int, int],
           max_cloud: float, platforms: list[str] | None) -> list:
    spec = SENSORS[sensor]
    aoi = grid.footprint_lonlat().__geo_interface__
    filters: list[dict] = [{"op": "<=", "args": [{"property": "eo:cloud_cover"}, max_cloud]}]
    if platforms:
        filters.append({"op": "in", "args": [{"property": "platform"}, platforms]})
    items = []
    for year in years:
        start = date(year, months[0], 1)
        end = date(year + (months[1] == 12), months[1] % 12 + 1, 1)
        found = _catalog().search(
            collections=[spec["collection"]],
            intersects=aoi,
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            filter={"op": "and", "args": filters},
            filter_lang="cql2-json",
        ).item_collection()
        items.extend(found)
    items.sort(key=lambda i: i.datetime)
    return items


def _clear_mask(sensor: str, mask: np.ndarray) -> np.ndarray:
    if sensor == "landsat":
        return (mask & LANDSAT_BAD_BITS) == 0
    return np.isin(mask, S2_CLEAR_CLASSES)


def _reflectance(sensor: str, dn: np.ndarray, items_by_day: dict, days: np.ndarray) -> np.ndarray:
    """Digital numbers (time, H, W) -> surface reflectance."""
    dn = dn.astype(np.float32)
    if sensor == "landsat":
        return dn * 2.75e-5 - 0.2
    # Processing baseline 04.00+ (from 2022-01-25) adds a +1000 offset to L2A values.
    offsets = np.array([
        -1000.0 if items_by_day[str(d)[:10]].properties.get("s2:processing_baseline", "00.00") >= "04.00" else 0.0
        for d in days
    ], dtype=np.float32)
    return (dn + offsets[:, None, None]) / 10000.0


# Per pixel, keep the clear view whose blue reflectance sits at this rank (0 = darkest).
# Cloud masks miss thin and bright cloud, and missed cloud is nearly always brighter than the
# ground in blue, so taking a view from the darker part of the stack rejects it. Using one
# whole view (not per-band statistics) keeps colours and NDVI physically consistent.
DARK_RANK = 0.25


def _darker_quartile_index(blue: np.ndarray, valid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    order = np.argsort(np.where(valid, blue, np.inf), axis=0)  # invalid views sort last
    rank = np.clip(((counts - 1) * DARK_RANK).astype(np.int64), 0, None)
    return np.take_along_axis(order, rank[None], axis=0)[0]


def quality_mosaic(sensor: str, grid: SiteGrid, items: list, bands: list[str],
                   blue: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Cloud-masked darker-quartile mosaic of `bands`; returns (band -> (H, W) reflectance, clear counts)."""
    if not items:
        raise RuntimeError(f"no {sensor} scenes matched; widen years/months or max_scene_cloud")
    spec = SENSORS[sensor]
    ds = load(
        items,
        bands=[*bands, spec["mask_band"]],
        geobox=grid.geobox,
        groupby="solar_day",
        resampling={"*": "bilinear", spec["mask_band"]: "nearest"},
        fail_on_error=False,
        pool=16,  # parallel COG reads; sequential loading is network-bound and slow
    )
    items_by_day = {i.datetime.date().isoformat(): i for i in items}
    days = ds.time.values
    clear = _clear_mask(sensor, ds[spec["mask_band"]].values)
    layers = []
    for name in bands:
        dn = ds[name].values
        refl = _reflectance(sensor, dn, items_by_day, days)
        refl[~clear | (dn == 0)] = np.nan
        layers.append(refl)
    del ds
    stack = np.stack(layers, axis=1)  # (time, band, H, W)
    del layers
    valid = np.isfinite(stack).all(axis=1)
    counts = valid.sum(axis=0)
    pick = _darker_quartile_index(stack[:, bands.index(blue)], valid, counts)
    chosen = np.take_along_axis(stack, pick[None, None], axis=0)[0]  # (band, H, W)
    chosen[:, counts == 0] = np.nan
    return {name: chosen[i].astype(np.float32) for i, name in enumerate(bands)}, counts


def composite(sensor: str, grid: SiteGrid, items: list) -> Composite:
    red_b, green_b, blue_b, nir_b = SENSORS[sensor]["bands"]
    mosaic, counts = quality_mosaic(sensor, grid, items, SENSORS[sensor]["bands"], blue=blue_b)
    red, green, blue, nir = mosaic[red_b], mosaic[green_b], mosaic[blue_b], mosaic[nir_b]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = ((nir - red) / (nir + red)).astype(np.float32)
        ndwi = ((green - nir) / (green + nir)).astype(np.float32)
    return Composite(
        rgb=np.stack([red, green, blue]),
        ndvi=ndvi,
        ndwi=ndwi,
        nir=nir,
        sensor=sensor,
        platforms=sorted({i.properties["platform"] for i in items}),
        scene_ids=[i.id for i in items],
        scene_dates=sorted({i.datetime.date().isoformat() for i in items}),
        clear_fraction=float((counts > 0).mean()),
        median_clear_obs=float(np.median(counts)),
    )


def platform_label(sensor: str, platforms: list[str]) -> str:
    if sensor == "sentinel-2":
        return "Sentinel-2"
    return " / ".join(PLATFORM_NAMES.get(p, p) for p in platforms)
