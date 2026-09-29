"""Mining classifier: finds open ground that looks like mine ground (see ml_regions.yaml)."""

from pathlib import Path

import yaml

from .features import Region

PIPELINE = Path(__file__).resolve().parents[2]
ML_DATA = PIPELINE / "data" / "ml"  # feature caches, models, candidates (not committed)
HANSEN = PIPELINE / "data" / "raw" / "hansen"
TANG_WERNER = PIPELINE / "data" / "raw" / "tang_werner" / "tw.shp"
REPORT = PIPELINE / "ML_REPORT.md"  # committed: the honest scorecard


def regions() -> dict[str, dict[str, Region]]:
    cfg = yaml.safe_load((PIPELINE / "ml_regions.yaml").read_text())
    d = cfg["defaults"]
    out: dict[str, dict[str, Region]] = {}
    for split in ("train", "test"):
        out[split] = {
            name: Region(name=name, center=tuple(r["center"]), box_km=r["box_km"],
                         pixel_m=r.get("pixel_m", d["pixel_m"]), months=tuple(d["months"]),
                         max_scene_cloud=r.get("max_scene_cloud", d["max_scene_cloud"]))
            for name, r in cfg[split].items()
        }
    return out
