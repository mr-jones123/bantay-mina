"""Build before/after imagery and forest-loss figures for the website.

    uv run python -m bantay build didipio
    uv run python -m bantay build --all
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import footprint as fp_mod
from .change import vegetation_to_bare, water_change
from .forest import forest_loss, land_mask
from .grid import site_grid
from .imagery import SENSORS, Composite, composite, platform_label, search
from .render import loss_overlay_png, to_rgb8, write_geotiff, write_webp

PIPELINE = Path(__file__).resolve().parent.parent
ROOT = PIPELINE.parent
PUBLIC = ROOT / "site" / "public" / "sites"
ANALYSIS = ROOT / "site" / "src" / "data" / "analysis"
RAW = PIPELINE / "data" / "raw"
TANG_WERNER = RAW / "tang_werner" / "tw.shp"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# Burned-in labels use ASCII hyphens: Pillow's built-in font has no en dash.
def _years_label(years: list[int]) -> str:
    return str(years[0]) if len(years) == 1 else f"{min(years)}-{max(years)}"


def _months_label(months: list[int]) -> str:
    return f"{MONTHS[months[0] - 1]}-{MONTHS[months[1] - 1]}"


def build_side(side: str, cmp: dict, cfg: dict, grid, out_dir: Path, web_dir: str) -> tuple[dict, Composite]:
    sensor = cmp["sensor"]
    spec = SENSORS[sensor]
    want = cmp[side]
    months = cfg["months"]
    max_cloud = cmp.get("max_scene_cloud", cfg["max_scene_cloud"])
    items = search(sensor, grid, want["years"], tuple(months), max_cloud, want.get("platforms"))
    print(f"  {cmp['id']} {side}: {len(items)} scenes")
    comp = composite(sensor, grid, items)
    rgb8 = to_rgb8(comp.rgb, tuple(cfg["stretch"]), cfg["gamma"])
    platforms = platform_label(sensor, comp.platforms)
    when = f"{_years_label(want['years'])}"
    lines = [
        f"{side.upper()} · {_months_label(months)} {when}",
        f"{platforms} · composite of {len(comp.scene_dates)} dates · {spec['native_m']} m",
        spec["attribution"],
    ]
    write_webp(rgb8, grid, lines, "left" if side == "before" else "right", out_dir / f"{side}.webp")
    write_geotiff(rgb8, grid, out_dir / f"{side}.tif", " | ".join(lines))
    meta = {
        "years": want["years"],
        "months": months,
        "platforms": comp.platforms,
        "platformLabel": platforms,
        "sceneCount": len(comp.scene_ids),
        "sceneDates": comp.scene_dates,
        "sceneIds": comp.scene_ids,
        "clearFraction": round(comp.clear_fraction, 4),
        "medianClearObservations": comp.median_clear_obs,
        "image": f"{web_dir}/{side}.webp",
        "geotiff": f"{web_dir}/{side}.tif",
    }
    return meta, comp


def build(slug: str, config: dict) -> None:
    site = config["sites"][slug]
    cfg = {**config["defaults"], **{k: v for k, v in site.items() if k in config["defaults"]}}
    grid = site_grid(tuple(site["center"]), cfg["box_km"], cfg["pixel_m"])
    print(f"{slug}: {grid.width}x{grid.height} px @ {grid.pixel_m} m, {grid.crs.name}")
    site_public = PUBLIC / slug

    footprint = fp_mod.load(grid, TANG_WERNER)
    footprint_out = None
    if footprint:
        fp_mod.write_geojson(footprint, site_public / "footprint.geojson")
        footprint_out = {
            "source": "Tang & Werner 2023",
            "citation": fp_mod.CITATION,
            "url": fp_mod.ZENODO,
            "polygons": footprint.polygons,
            "areaHa": round(footprint.utm.area / 10_000, 1),
            "svgPath": fp_mod.svg_path(footprint, grid),
            "geojson": f"/sites/{slug}/footprint.geojson",
        }
        print(f"  footprint: {footprint.polygons} polygons, {footprint_out['areaHa']} ha")

    forest, loss_grid = forest_loss(grid, footprint.lonlat if footprint else None, RAW / "hansen")
    first, last = forest["years"]
    loss_overlay_png(loss_grid, first, last, site_public / "forest-loss.png")
    forest["overlay"] = f"/sites/{slug}/forest-loss.png"
    land = land_mask(grid, RAW / "hansen")
    print(f"  forest loss in box: {forest['box']['lossHa']} ha")

    comparisons = []
    for cmp in site["comparisons"]:
        web_dir = f"/sites/{slug}/{cmp['id']}"
        out_dir = site_public / cmp["id"]
        before, before_comp = build_side("before", cmp, cfg, grid, out_dir, web_dir)
        after, after_comp = build_side("after", cmp, cfg, grid, out_dir, web_dir)
        fp_utm = footprint.utm if footprint else None
        change = vegetation_to_bare(before_comp.ndvi, after_comp.ndvi, land, grid, fp_utm, out_dir / "change.png")
        change["overlay"] = f"{web_dir}/change.png"
        water = water_change(before_comp.ndwi, before_comp.nir, after_comp.ndwi, after_comp.nir,
                             land, grid, fp_utm, out_dir / "water.png")
        water["overlay"] = f"{web_dir}/water.png"
        print(f"  {cmp['id']}: vegetation -> bare {change['vegetationToBareHa']} ha, "
              f"land -> water {water['landToWaterHa']} ha, water -> land {water['waterToLandHa']} ha")
        comparisons.append({
            "id": cmp["id"],
            "title": cmp["title"],
            "sensor": cmp["sensor"],
            "nativeResolutionM": SENSORS[cmp["sensor"]]["native_m"],
            "attribution": SENSORS[cmp["sensor"]]["attribution"],
            "before": before,
            "after": after,
            "change": change,
            "water": water,
        })

    b = grid.footprint_lonlat().bounds
    analysis = {
        "slug": slug,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "center": site["center"],
        "boxKm": cfg["box_km"],
        "bounds": [round(v, 6) for v in b],
        "crs": f"EPSG:{grid.crs.to_epsg()}",
        "pixelM": grid.pixel_m,
        "sizePx": [grid.width, grid.height],
        "stretch": cfg["stretch"],
        "gamma": cfg["gamma"],
        "comparisons": comparisons,
        "footprint": footprint_out,
        "forest": forest,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / f"{slug}.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {ANALYSIS / f'{slug}.json'}")


def retitle(slug: str, config: dict) -> None:
    """Copy comparison titles from sites.yaml into an existing analysis JSON, without reprocessing."""
    path = ANALYSIS / f"{slug}.json"
    analysis = json.loads(path.read_text())
    titles = {c["id"]: c["title"] for c in config["sites"][slug]["comparisons"]}
    for cmp in analysis["comparisons"]:
        if cmp["id"] not in titles:
            raise SystemExit(f"{slug}: comparison {cmp['id']!r} no longer in sites.yaml; run build instead")
        cmp["title"] = titles[cmp["id"]]
    path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    print(f"{slug}: titles updated")


def clouds(slug: str, config: dict) -> None:
    """Score month windows for a site from the Sentinel-2 cloud record (see bantay/clouds.py)."""
    from .clouds import report

    site = config["sites"][slug]
    cfg = {**config["defaults"], **{k: v for k, v in site.items() if k in config["defaults"]}}
    report(slug, site_grid(tuple(site["center"]), cfg["box_km"], cfg["pixel_m"]), tuple(cfg["months"]))


def main() -> None:
    parser = argparse.ArgumentParser(prog="bantay")
    sub = parser.add_subparsers(dest="cmd", required=True)
    commands = {
        "build": (build, "build imagery + stats for one or all sites"),
        "retitle": (retitle, "refresh comparison titles in built JSON from sites.yaml"),
        "clouds": (clouds, "rank month windows by cloud-free coverage, to choose a site's `months`"),
    }
    for name, (_, help_text) in commands.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("slug", nargs="?")
        p.add_argument("--all", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((PIPELINE / "sites.yaml").read_text())
    slugs = list(config["sites"]) if args.all else [args.slug]
    if not slugs or slugs == [None]:
        parser.error("give a site slug or --all")
    command = commands[args.cmd][0]
    for slug in slugs:
        if slug not in config["sites"]:
            parser.error(f"unknown site {slug!r}; known: {', '.join(config['sites'])}")
        command(slug, config)


if __name__ == "__main__":
    main()
