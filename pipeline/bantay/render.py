"""Turn composites and overlays into web images with provenance burned in."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

from .grid import SiteGrid

NO_DATA_GREY = 96


def to_rgb8(rgb: np.ndarray, stretch: tuple[float, float], gamma: float) -> np.ndarray:
    """Identical linear stretch + gamma for every image of a site. NaN -> flat grey."""
    lo, hi = stretch
    scaled = np.clip((rgb - lo) / (hi - lo), 0, 1) ** (1 / gamma)
    out = np.nan_to_num(scaled * 255, nan=NO_DATA_GREY).astype(np.uint8)
    return np.moveaxis(out, 0, -1)  # (H, W, 3)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _label(draw: ImageDraw.ImageDraw, lines: list[str], corner: str, w: int, h: int) -> None:
    title_font, body_font = _font(max(14, w // 34)), _font(max(11, w // 56))
    fonts = [title_font] + [body_font] * (len(lines) - 1)
    pad, gap = w // 60, 3
    sizes = [draw.textbbox((0, 0), t, font=f) for t, f in zip(lines, fonts)]
    box_w = max(b[2] - b[0] for b in sizes) + 2 * pad
    box_h = sum(b[3] - b[1] for b in sizes) + gap * (len(lines) - 1) + 2 * pad
    margin = w // 50
    x0 = margin if corner == "left" else w - margin - box_w
    y0 = h - margin - box_h
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(0, 0, 0, 170))
    y = y0 + pad
    for text, font, b in zip(lines, fonts, sizes):
        draw.text((x0 + pad, y - b[1]), text, font=font, fill=(255, 255, 255, 255))
        y += b[3] - b[1] + gap


def _scale_bar(draw: ImageDraw.ImageDraw, grid: SiteGrid) -> None:
    w = grid.width
    km = 1 if grid.width * grid.pixel_m >= 3000 else 0.5
    length = km * 1000 / grid.pixel_m
    margin = w // 50
    x1, y = w - margin, margin + 18
    x0 = x1 - length
    draw.rectangle((x0 - 8, margin - 4, x1 + 8, y + 10), fill=(0, 0, 0, 150))
    draw.rectangle((x0, y, x1, y + 5), fill=(255, 255, 255, 255))
    font = _font(max(11, w // 56))
    text = f"{km:g} km"
    tb = draw.textbbox((0, 0), text, font=font)
    draw.text((x0 + (length - (tb[2] - tb[0])) / 2, margin - tb[1]), text, font=font, fill="white")


def write_webp(rgb8: np.ndarray, grid: SiteGrid, lines: list[str], corner: str, path: Path) -> None:
    img = Image.fromarray(rgb8, "RGB").convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    _label(draw, lines, corner, grid.width, grid.height)
    _scale_bar(draw, grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(img, overlay).convert("RGB").save(path, "WEBP", quality=88, method=6)


def write_geotiff(rgb8: np.ndarray, grid: SiteGrid, path: Path, description: str) -> None:
    """Unannotated, georeferenced copy for anyone who wants to check our work in QGIS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=grid.width, height=grid.height, count=3,
        dtype="uint8", crs=grid.crs.to_wkt(), transform=grid.geobox.transform,
        compress="deflate", tiled=True, photometric="RGB",
    ) as dst:
        dst.write(np.moveaxis(rgb8, -1, 0))
        dst.update_tags(DESCRIPTION=description)


# Hansen loss year 1..N -> colour ramp from yellow (early) to deep red (recent).
def loss_overlay_png(loss_year: np.ndarray, first: int, last: int, path: Path) -> None:
    h, w = loss_year.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    lost = loss_year > 0
    t = np.clip((loss_year.astype(np.float32) - first) / max(1, last - first), 0, 1)
    rgba[..., 0] = 255
    rgba[..., 1] = (220 * (1 - t)).astype(np.uint8)
    rgba[..., 2] = (40 * (1 - t)).astype(np.uint8)
    rgba[..., 3] = np.where(lost, 200, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)
