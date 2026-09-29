"""Train the mining classifier and write an honest scorecard.

Protocol
1. Training regions only: leave-one-region-out. Each fold trains on balanced pixel samples from the
   other regions and predicts *every* open-ground pixel of the held-out region, so precision is
   measured at the real (rare) share of mining, not at the 1-in-6 share of the balanced sample.
2. The pooled out-of-fold predictions choose the publishing threshold and fit an isotonic
   calibration of P(mining). Northern Luzon has not been touched yet.
3. The final model trains on all training regions and is scored once on the Northern Luzon test
   regions, pixel by pixel and mine by mine.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

from . import HANSEN, ML_DATA, TANG_WERNER
from .dataset import CLASSES, labels, matrix, mine_masks, open_ground, sample
from .features import FEATURES, Region, load_or_build

YEAR = 2024
PER_CLASS = 4000
# The threshold fills a *review queue*: every flag is checked by a person before anything is
# published, so a queue that is three-quarters right is useful. Publication needs the human check.
REVIEW_PRECISION = 0.75
MIN_PATCH_PX = 25  # 1 ha at 20 m
MINING = CLASSES.index("mining")


@dataclass
class RegionData:
    region: Region
    features: dict[str, np.ndarray]
    y: np.ndarray  # class per pixel, -1 unused
    inside: np.ndarray
    near: np.ndarray


def load_region(region: Region) -> RegionData:
    f = load_or_build(region, YEAR, ML_DATA, HANSEN)
    inside, near = mine_masks(region.grid(), TANG_WERNER)
    return RegionData(region, f, labels(f, inside, near), inside, near)


def make_model(kind: str):
    if kind == "lightgbm":
        # Shallow, heavily regularised trees: the test is a *different region*, so fitting the
        # training regions' quirks tightly (the first run did) costs accuracy where it matters.
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=400, learning_rate=0.03,
                                  num_leaves=15, min_child_samples=200, subsample=0.7, subsample_freq=1,
                                  colsample_bytree=0.6, reg_lambda=5.0, verbose=-1, n_jobs=8)
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=8, random_state=0)
    raise ValueError(kind)


def _fit(kind: str, parts: list[RegionData], rng: np.random.Generator):
    xs, ys = [], []
    for d in parts:
        idx = sample(d.y, PER_CLASS, rng)
        xs.append(matrix(d.features, FEATURES, idx))
        ys.append(d.y.ravel()[idx])
    x, y = np.concatenate(xs), np.concatenate(ys)
    model = make_model(kind)
    if kind == "random_forest":
        x = np.nan_to_num(x, nan=-999.0)
    model.fit(x, y)
    return model


def p_mining(model, kind: str, f: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """P(mining) for every pixel in `mask` (flat order)."""
    x = matrix(f, FEATURES, np.flatnonzero(mask.ravel()))
    if kind == "random_forest":
        x = np.nan_to_num(x, nan=-999.0)
    proba = model.predict_proba(x)
    col = list(model.classes_).index(MINING)
    return proba[:, col]


def _scored(d: RegionData, p: np.ndarray, cand: np.ndarray):
    """Truth for scoring: mining inside polygons vs look-alikes; the 300 m ring is ignored."""
    flat_inside = d.inside.ravel()[np.flatnonzero(cand.ravel())]
    flat_near = d.near.ravel()[np.flatnonzero(cand.ravel())]
    keep = flat_inside | ~flat_near
    return p[keep], flat_inside[keep]


def threshold_for(y: np.ndarray, p: np.ndarray, target: float) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y, p)
    ok = np.flatnonzero(precision[:-1] >= target)
    if ok.size:
        i = ok[np.argmax(recall[ok])]
    else:  # target not reachable: take the most precise point that still finds something
        i = int(np.argmax(precision[:-1]))
    return float(thresholds[i]), float(precision[i]), float(recall[i])


def mine_level(d: RegionData, flagged: np.ndarray) -> dict:
    """Per mapped mine (connected polygon area): was at least a quarter of its open ground flagged?"""
    cand = open_ground(d.features)
    lab, n = ndimage.label(d.inside)
    hit = found = 0
    for k in range(1, n + 1):
        region = (lab == k) & cand
        if region.sum() < MIN_PATCH_PX:
            continue
        found += 1
        hit += (flagged[region].mean() >= 0.25)
    return {"minesWithOpenGround": int(found), "minesDetected": int(hit)}


def unmapped_patches(d: RegionData, flagged: np.ndarray) -> dict:
    """Flagged patches of ≥1 ha that touch no mapped mine: candidates, or false alarms."""
    lab, n = ndimage.label(flagged, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    near_hits = ndimage.maximum(d.near, lab, index=np.arange(1, n + 1))
    big = sizes >= MIN_PATCH_PX
    new = big & (near_hits == 0)
    px_ha = d.region.pixel_m ** 2 / 10_000
    return {"patches": int(new.sum()), "areaHa": round(float(sizes[new].sum() * px_ha), 1)}


def cross_validate(kind: str, train: list[RegionData]) -> dict:
    rng = np.random.default_rng(0)
    ps, ys, per_region = [], [], {}
    for held in train:
        model = _fit(kind, [d for d in train if d is not held], rng)
        cand = open_ground(held.features)
        p, y = _scored(held, p_mining(model, kind, held.features, cand), cand)
        ps.append(p)
        ys.append(y)
        per_region[held.region.name] = {
            "averagePrecision": round(float(average_precision_score(y, p)), 3) if y.any() else None,
            "miningShareOfOpenGround": round(float(y.mean()), 4),
        }
        print(f"  [{kind}] held out {held.region.name}: AP {per_region[held.region.name]['averagePrecision']}")
    p_all, y_all = np.concatenate(ps), np.concatenate(ys)
    t, prec, rec = threshold_for(y_all, p_all, REVIEW_PRECISION)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_all, y_all)
    return {"perRegion": per_region, "averagePrecision": round(float(average_precision_score(y_all, p_all)), 3),
            "threshold": t, "precision": round(prec, 3), "recall": round(rec, 3), "calibration": iso}


def evaluate(kind: str, model, threshold: float, test: list[RegionData]) -> dict:
    ps, ys, regions = [], [], {}
    for d in test:
        cand = open_ground(d.features)
        p_flat = p_mining(model, kind, d.features, cand)
        p, y = _scored(d, p_flat, cand)
        ps.append(p)
        ys.append(y)
        flagged = np.zeros(cand.shape, dtype=bool)
        flagged.ravel()[np.flatnonzero(cand.ravel())] = p_flat >= threshold
        sel = p >= threshold
        regions[d.region.name] = {
            "averagePrecision": round(float(average_precision_score(y, p)), 3) if y.any() else None,
            "precision": round(float(y[sel].mean()), 3) if sel.any() else None,
            "recall": round(float(sel[y].mean()), 3) if y.any() else None,
            **mine_level(d, flagged),
            "unmappedFlagged": unmapped_patches(d, flagged),
        }
    p_all, y_all = np.concatenate(ps), np.concatenate(ys)
    sel = p_all >= threshold
    return {"perRegion": regions,
            "averagePrecision": round(float(average_precision_score(y_all, p_all)), 3),
            "precision": round(float(y_all[sel].mean()), 3) if sel.any() else None,
            "recall": round(float(sel[y_all].mean()), 3),
            "precisionAtRecall": precision_at_recall(y_all, p_all, (0.25, 0.5, 0.75))}


def precision_at_recall(y: np.ndarray, p: np.ndarray, levels) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(y, p)
    return {f"{lvl:.2f}": round(float(precision[recall >= lvl].max()), 3) for lvl in levels}


def shap_ranking(model, test: list[RegionData], n: int = 3000) -> list[tuple[str, float]]:
    import shap
    rng = np.random.default_rng(1)
    xs = []
    for d in test:
        cand = np.flatnonzero((open_ground(d.features) & d.inside).ravel())
        other = np.flatnonzero((open_ground(d.features) & ~d.near).ravel())
        pick = np.concatenate([rng.choice(cand, min(n // 10, cand.size), replace=False) if cand.size else cand,
                               rng.choice(other, min(n // 10, other.size), replace=False)])
        xs.append(matrix(d.features, FEATURES, pick))
    x = np.concatenate(xs)
    values = shap.TreeExplainer(model).shap_values(x)
    arr = np.asarray(values)
    # shap returns (classes, n, f) for older versions or (n, f, classes) for newer ones
    mining = arr[MINING] if arr.shape[0] == len(CLASSES) and arr.ndim == 3 and arr.shape[1] == len(x) else arr[..., MINING]
    importance = np.abs(mining).mean(axis=0)
    order = np.argsort(importance)[::-1]
    return [(FEATURES[i], round(float(importance[i]), 4)) for i in order]


def run(regions: dict[str, dict[str, Region]], report: Path) -> None:
    print("loading training regions")
    train = [load_region(r) for r in regions["train"].values()]
    print("loading test regions (Northern Luzon, held out)")
    test = [load_region(r) for r in regions["test"].values()]

    results = {}
    for kind in ("lightgbm", "random_forest"):
        print(f"cross-validating {kind}")
        cv = cross_validate(kind, train)
        model = _fit(kind, train, np.random.default_rng(42))
        results[kind] = {"cv": {k: v for k, v in cv.items() if k != "calibration"},
                         "test": evaluate(kind, model, cv["threshold"], test),
                         "model": model, "calibration": cv["calibration"]}

    best = results["lightgbm"]
    out = ML_DATA / "model"
    out.mkdir(parents=True, exist_ok=True)
    best["model"].booster_.save_model(str(out / "lightgbm.txt"))
    joblib.dump({"model": best["model"], "calibration": best["calibration"],
                 "threshold": best["cv"]["threshold"], "features": FEATURES, "classes": CLASSES,
                 "year": YEAR}, out / "model.joblib")
    ranking = shap_ranking(best["model"], test)
    metrics = {k: {"cv": v["cv"], "test": v["test"]} for k, v in results.items()}
    (out / "metrics.json").write_text(json.dumps({**metrics, "shap": ranking}, indent=2))
    write_report(report, metrics, ranking, regions)
    print(f"wrote {report}")


def write_report(path: Path, m: dict, ranking: list, regions: dict) -> None:
    lg, rf = m["lightgbm"], m["random_forest"]
    lines = [
        "# Mining classifier — scorecard",
        "",
        "Generated by `uv run python -m bantay.ml train`. Pixel scores are over **all open ground** in",
        "each region (not a balanced sample), so precision reflects how rare mine ground really is.",
        f"Features: {YEAR} dry-season Sentinel-2/Sentinel-1 at 20 m; labels: Tang & Werner 2023 mine",
        "polygons vs ESA WorldCover look-alikes more than 300 m from any mapped mine.",
        "",
        f"Training regions ({len(regions['train'])}): " + ", ".join(regions["train"]),
        f"Held-out test regions, Northern Luzon ({len(regions['test'])}): " + ", ".join(regions["test"]),
        "",
        "## Headline",
        "",
        f"| Model | CV avg. precision | Review threshold (CV, ≥{REVIEW_PRECISION:.0%} precision) | CV precision / recall "
        "| Test avg. precision | Test precision / recall at that threshold | Test precision at 25% / 50% / 75% recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in (("LightGBM", lg), ("Random forest", rf)):
        cv, t = r["cv"], r["test"]
        lines.append(f"| {name} | {cv['averagePrecision']} | {cv['threshold']:.3f} | {cv['precision']} / {cv['recall']} "
                     f"| {t['averagePrecision']} | {t['precision']} / {t['recall']} "
                     f"| {' / '.join(str(v) for v in t['precisionAtRecall'].values())} |")
    lines += ["", "## Northern Luzon, region by region (LightGBM)", "",
              "| Region | Avg. precision | Precision | Recall | Mapped mines found | Flagged patches outside mapped mines (≥1 ha) |",
              "|---|---|---|---|---|---|"]
    for name, r in lg["test"]["perRegion"].items():
        u = r["unmappedFlagged"]
        lines.append(f"| {name} | {r['averagePrecision']} | {r['precision']} | {r['recall']} | "
                     f"{r['minesDetected']} of {r['minesWithOpenGround']} | {u['patches']} ({u['areaHa']} ha) |")
    lines += ["", "## Leave-one-region-out (training regions, LightGBM)", "",
              "| Held-out region | Avg. precision | Mining share of open ground |", "|---|---|---|"]
    for name, r in lg["cv"]["perRegion"].items():
        lines.append(f"| {name} | {r['averagePrecision']} | {r['miningShareOfOpenGround']} |")
    lines += ["", "## What the model relies on (mean |SHAP| for the mining class, test regions)", "",
              "| Feature | Mean \\|SHAP\\| |", "|---|---|"]
    lines += [f"| `{name}` | {v} |" for name, v in ranking[:15]]
    lines += ["", "## How to read this", "",
              "- *Average precision* summarises the whole precision–recall curve; the share of mine ground",
              "  among open ground is the score a random guess would get.",
              "- The threshold is fixed on training regions only, then applied unchanged to Northern Luzon.",
              "- \"Flagged patches outside mapped mines\" are **unreviewed candidates**: some will be real",
              "  mines newer or smaller than the 2023 map, some quarries, some false alarms. None is published",
              "  without a person checking it.", ""]
    path.write_text("\n".join(lines))
