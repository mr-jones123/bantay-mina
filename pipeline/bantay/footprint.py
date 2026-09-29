"""Mine footprint polygons from Tang & Werner (2023), clipped to a site's analysis box."""

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from shapely import force_2d
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.geometry.base import BaseGeometry

from .grid import SiteGrid

ZENODO = "https://zenodo.org/records/7894216"
CITATION = (
    "Tang, L. & Werner, T. T. 2023. Global mining footprint mapped from high-resolution satellite "
    "imagery. Communications Earth & Environment 4: 134. doi:10.1038/s43247-023-00805-6 (CC BY 4.0)."
)


@dataclass
class Footprint:
    utm: BaseGeometry  # clipped union in the site grid CRS
    lonlat: BaseGeometry
    polygons: int


def load(grid: SiteGrid, shapefile: Path) -> Footprint | None:
    if not shapefile.exists():
        raise FileNotFoundError(
            f"{shapefile} missing — download the Tang & Werner shapefile from {ZENODO} (see README)"
        )
    box_utm = gpd.GeoSeries([grid.footprint()], crs=grid.crs)
    found = gpd.read_file(shapefile, bbox=box_utm)
    if found.empty:
        return None
    utm = found.to_crs(grid.crs)
    clipped = utm.geometry.intersection(grid.footprint())
    clipped = clipped[~clipped.is_empty & (clipped.area > 0)]
    if clipped.empty:
        return None
    union = force_2d(clipped.union_all().buffer(0))  # source polygons carry Z values
    return Footprint(
        utm=union,
        lonlat=gpd.GeoSeries([union], crs=grid.crs).to_crs(4326).iloc[0],
        polygons=len(clipped),
    )


def _polygons(geom: BaseGeometry) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for part in getattr(geom, "geoms", []) for g in _polygons(part)]


def svg_path(fp: Footprint, grid: SiteGrid) -> str:
    """Outline in image-pixel coordinates, for an SVG overlay with viewBox 0 0 W H."""
    parts = []
    for poly in _polygons(fp.utm):
        for ring in [poly.exterior, *poly.interiors]:
            pts = [grid.to_pixel(x, y) for x, y in ring.coords]
            parts.append("M" + "L".join(f"{c:.1f},{r:.1f}" for c, r in pts) + "Z")
    return "".join(parts)


def write_geojson(fp: Footprint, path: Path) -> None:
    feature = {
        "type": "Feature",
        "properties": {"source": "Tang & Werner 2023", "polygons": fp.polygons},
        "geometry": mapping(fp.lonlat.simplify(0.00002)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
