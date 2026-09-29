"""The single pixel grid every layer of a site is rendered onto."""

from dataclasses import dataclass

from odc.geo.geobox import GeoBox
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box
from shapely.ops import transform


@dataclass(frozen=True)
class SiteGrid:
    geobox: GeoBox
    crs: CRS
    pixel_m: float

    @property
    def width(self) -> int:
        return self.geobox.shape.x

    @property
    def height(self) -> int:
        return self.geobox.shape.y

    def footprint(self) -> Polygon:
        """Analysis box in the grid CRS (metres)."""
        b = self.geobox.boundingbox
        return box(b.left, b.bottom, b.right, b.top)

    def footprint_lonlat(self) -> Polygon:
        """Analysis box in WGS84; slightly non-rectangular because the grid is UTM."""
        to_ll = Transformer.from_crs(self.crs, 4326, always_xy=True).transform
        return transform(to_ll, self.footprint().segmentize(100))

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Grid-CRS coordinate -> fractional (column, row) in the output image."""
        b = self.geobox.boundingbox
        return (x - b.left) / self.pixel_m, (b.top - y) / self.pixel_m


def utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def site_grid(center: tuple[float, float], box_km: float, pixel_m: float) -> SiteGrid:
    lon, lat = center
    crs = utm_crs(lon, lat)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    half = box_km * 500
    # Snap the box to whole pixels so every layer shares identical edges.
    left = round((x - half) / pixel_m) * pixel_m
    bottom = round((y - half) / pixel_m) * pixel_m
    size = round(box_km * 1000 / pixel_m) * pixel_m
    geobox = GeoBox.from_bbox(
        (left, bottom, left + size, bottom + size), crs=crs.to_epsg(), resolution=pixel_m
    )
    return SiteGrid(geobox=geobox, crs=crs, pixel_m=pixel_m)
