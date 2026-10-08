"""Legally protected areas (pipeline/protected_areas.yaml), measured against each site.

An area's boundary comes either from the corner coordinates printed in its law, or, where the law
gives none, from the Biodiversity Management Bureau's own polygon (cached under
pipeline/protected_areas/). Either way it is checked against the area the law states. For a site,
the pipeline measures how far the mapped mine footprint is from the boundary, and how much clearing
falls inside the area and inside its buffer zone (if the law sets one).
"""

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import yaml
from rasterio.features import geometry_mask
from shapely.geometry import LineString, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry

from .grid import SiteGrid

AREA_TOLERANCE = 0.03  # the boundary must enclose the stated area to within 3%
# Public map service of Geoportal Philippines (NAMRIA); BMB's NIPAS layer answers point queries
# with the polygon itself.
BMB_WMS = "https://geoserver.geoportal.gov.ph/geoserver/wms"
BMB_LAYER = "geoportal:bmb_protectedareas"


@dataclass(frozen=True)
class ProtectedArea:
    id: str
    name: str
    short: str
    law: str
    url: str
    buffer_km: float  # 0 when the law sets no buffer zone
    lonlat: BaseGeometry


def _degrees(dms: str) -> float:
    d, m, s = (int(v) for v in dms.split("-"))
    return d + m / 60 + s / 3600


def _area_ha(geom: BaseGeometry) -> float:
    s = gpd.GeoSeries([geom], crs=4326)
    return float(s.to_crs(s.estimate_utm_crs()).area.iloc[0] / 1e4)


def _from_corners(spec: dict) -> BaseGeometry:
    fixes = spec.get("corrections", {})
    ring = []
    for n, lon, lat in spec["corners"]:
        fix = fixes.get(n, {})
        ring.append((_degrees(fix.get("longitude", lon)), _degrees(fix.get("latitude", lat))))
    # Coordinates in older laws use the Luzon 1911 datum; move them onto WGS84 like every other layer.
    return gpd.GeoSeries([Polygon(ring)], crs=spec.get("datum", 4326)).to_crs(4326).iloc[0]


def fetch_bmb(lon: float, lat: float, paname: str) -> dict:
    """BMB's polygon for the protected area named `paname` at a point inside it."""
    d = 0.005
    query = (f"{BMB_WMS}?service=WMS&version=1.1.1&request=GetFeatureInfo&layers={BMB_LAYER}"
             f"&query_layers={BMB_LAYER}&styles=&srs=EPSG:4326&bbox={lon - d},{lat - d},{lon + d},{lat + d}"
             "&width=101&height=101&x=50&y=50&info_format=application/json&feature_count=20")
    with urllib.request.urlopen(query, timeout=60) as r:
        features = json.load(r)["features"]
    match = [f for f in features if f["properties"].get("paname") == paname]
    if len(match) != 1:
        raise ValueError(f"BMB layer: {len(match)} areas named {paname!r} at {lon}, {lat}")
    return match[0]


def _from_bmb(area_id: str, spec: dict, cache_dir: Path) -> BaseGeometry:
    path = cache_dir / f"{area_id}.geojson"
    if not path.exists():
        bmb = spec["bmb"]
        feature = fetch_bmb(*bmb["point"], bmb["paname"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(feature, ensure_ascii=False) + "\n")
    return shape(json.loads(path.read_text())["geometry"]).buffer(0)


def load(path: Path) -> list[ProtectedArea]:
    areas = []
    for area_id, spec in yaml.safe_load(path.read_text()).items():
        geom = _from_corners(spec) if "corners" in spec else _from_bmb(area_id, spec, path.with_suffix(""))
        if not geom.is_valid:
            raise ValueError(f"{area_id}: boundary crosses itself; check the corners")
        got, want = _area_ha(geom), spec["stated_area_ha"]
        if abs(got - want) / want > AREA_TOLERANCE:
            raise ValueError(f"{area_id}: boundary encloses {got:,.0f} ha, law states {want:,} ha")
        areas.append(ProtectedArea(area_id, spec["name"], spec["short"], spec["law"], spec["url"],
                                   spec.get("buffer_km") or 0, geom))
    return areas


def write_geojson(areas: list[ProtectedArea], path: Path) -> None:
    """Area and buffer-zone outlines for the site maps."""
    features = []
    for a in areas:
        props = {"id": a.id, "name": a.name, "law": a.law, "url": a.url}
        features.append({"type": "Feature", "properties": {**props, "kind": "area"},
                         "geometry": mapping(a.lonlat)})
        if a.buffer_km:
            s = gpd.GeoSeries([a.lonlat], crs=4326)
            outer = s.to_crs(s.estimate_utm_crs()).buffer(a.buffer_km * 1000).boundary.to_crs(4326).iloc[0]
            features.append({"type": "Feature", "properties": {**props, "kind": "buffer", "bufferKm": a.buffer_km},
                             "geometry": mapping(outer.simplify(0.0002))})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def _lines_svg(geom: BaseGeometry, grid: SiteGrid) -> str:
    lines = [geom] if isinstance(geom, LineString) else list(getattr(geom, "geoms", []))
    parts = []
    for line in lines:
        if not isinstance(line, LineString) or line.is_empty:
            continue
        pts = [grid.to_pixel(x, y) for x, y in line.coords]
        parts.append("M" + "L".join(f"{c:.1f},{r:.1f}" for c, r in pts))
    return "".join(parts)


def _poly_svg(geom: BaseGeometry, grid: SiteGrid) -> str:
    polys = [geom] if isinstance(geom, Polygon) else [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    parts = []
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            pts = [grid.to_pixel(x, y) for x, y in ring.coords]
            parts.append("M" + "L".join(f"{c:.1f},{r:.1f}" for c, r in pts) + "Z")
    return "".join(parts)


def _mask(geom: BaseGeometry, grid: SiteGrid) -> np.ndarray:
    if geom.is_empty:
        return np.zeros((grid.height, grid.width), dtype=bool)
    return geometry_mask([geom], out_shape=(grid.height, grid.width), transform=grid.geobox.transform, invert=True)


def for_site(areas: list[ProtectedArea], grid: SiteGrid, footprint_utm: BaseGeometry | None,
             changes: dict[str, np.ndarray]) -> list[dict]:
    """Areas whose boundary or buffer zone reaches the analysis box, measured against the site.

    `changes` maps comparison id -> vegetation-to-bare mask on the site grid.
    """
    box = grid.footprint()
    px_ha = grid.pixel_m**2 / 10_000
    out = []
    for a in areas:
        area = gpd.GeoSeries([a.lonlat], crs=4326).to_crs(grid.crs).iloc[0]
        outer = area.buffer(a.buffer_km * 1000) if a.buffer_km else area
        if not outer.intersects(box):
            continue
        ring = area.boundary.intersection(box)
        buffer_ring = outer.boundary.intersection(box) if a.buffer_km else LineString()
        inside, in_buffer = _mask(area.intersection(box), grid), _mask(outer.difference(area).intersection(box), grid)
        entry = {
            "id": a.id,
            "name": a.name,
            "short": a.short,
            "law": a.law,
            "url": a.url,
            "bufferKm": a.buffer_km,
            "areaSvgPath": _poly_svg(area.intersection(box), grid),
            "boundarySvgPath": _lines_svg(ring, grid),
            "bufferSvgPath": _lines_svg(buffer_ring, grid),
            "footprintDistanceKm": None,
            "footprintInsideHa": None,
            "footprintInBufferHa": None,
            "change": {
                cmp_id: {
                    "insideHa": round(float((mask & inside).sum() * px_ha), 1),
                    "bufferHa": round(float((mask & in_buffer).sum() * px_ha), 1),
                }
                for cmp_id, mask in changes.items()
            },
        }
        if footprint_utm is not None:
            entry["footprintDistanceKm"] = round(footprint_utm.distance(area) / 1000, 2)
            entry["footprintInsideHa"] = round(footprint_utm.intersection(area).area / 1e4, 1)
            entry["footprintInBufferHa"] = round(footprint_utm.intersection(outer.difference(area)).area / 1e4, 1)
        out.append(entry)
    return out
