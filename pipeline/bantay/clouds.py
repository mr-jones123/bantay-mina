"""Which months give clear composites at a site? Measured from the Sentinel-2 cloud record.

For every Sentinel-2 date over the site box (2019-2025, plus 2026 so far), the scene classification
layer gives each pixel as clear or not. A run of months is scored by the share of the box that had at
least three clear views in that window, in the *cloudiest* of the seven full years: a composite needs
several clear views per pixel, and the window has to work every year, not on average.
"""

from datetime import date

import numpy as np
from odc.stac import load

from .grid import SiteGrid
from .imagery import S2_CLEAR_CLASSES, _catalog

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
YEARS = range(2019, 2026)  # full years used for scoring
PROBE_M = 60  # the cloud mask is a 20 m product; 60 m is plenty to measure cover
MIN_VIEWS = 3


def clear_record(grid: SiteGrid, last_year: int) -> dict[str, np.ndarray]:
    """Date (YYYY-MM-DD) -> boolean clear mask on a coarse copy of the site grid."""
    coarse = grid.geobox.zoom_to(resolution=PROBE_M)
    aoi = grid.footprint_lonlat().__geo_interface__
    record: dict[str, np.ndarray] = {}
    for year in range(YEARS.start, last_year + 1):
        items = _catalog().search(collections=["sentinel-2-l2a"], intersects=aoi,
                                  datetime=f"{year}-01-01/{year}-12-31").item_collection()
        if not items:
            continue
        ds = load(items, bands=["SCL"], geobox=coarse, groupby="solar_day",
                  resampling="nearest", fail_on_error=False, pool=16)
        for t, scl in zip(ds.time.values, ds["SCL"].values):
            if (scl > 0).mean() < 0.5:  # swath edge: mostly no data
                continue
            record[str(t)[:10]] = np.isin(scl, S2_CLEAR_CLASSES)
        print(f"  {year}: {len(items)} scenes", flush=True)
    return record


def _coverage(record: dict[str, np.ndarray], year: int, months: tuple[int, int]) -> float | None:
    days = [d for d in record if int(d[:4]) == year and months[0] <= int(d[5:7]) <= months[1]]
    if not days:
        return None
    views = np.sum([record[d] for d in days], axis=0)
    return float((views >= MIN_VIEWS).mean())


def report(slug: str, grid: SiteGrid, current: tuple[int, int]) -> None:
    today = date.today()
    record = clear_record(grid, today.year)
    print(f"{slug}: {len(record)} Sentinel-2 dates")
    print("  mean clear share by month (2019-2025):")
    print("   " + " ".join(f"{m:>4}" for m in MONTHS))
    shares = [[float(v.mean()) for d, v in record.items() if int(d[5:7]) == m and int(d[:4]) in YEARS]
              for m in range(1, 13)]
    print("   " + " ".join(f"{100 * np.mean(s):4.0f}" if s else "   -" for s in shares))

    rows = []
    for length in (3, 4, 5, 6):
        for a in range(1, 14 - length):
            window = (a, a + length - 1)
            past = [_coverage(record, y, window) for y in YEARS]
            if any(c is None for c in past):
                continue
            complete = window[1] < today.month  # this year's window already over?
            rows.append((min(past), float(np.mean(past)), window,
                         _coverage(record, today.year, window) if complete else None))
    rows.sort(key=lambda r: (-round(r[0], 2), r[2][1] - r[2][0]))

    def line(r: tuple) -> str:
        worst, mean, (a, b), now = r
        now_txt = f"{today.year}: {now:.0%}" if now is not None else f"{today.year}: not over yet"
        return f"   {MONTHS[a - 1]}-{MONTHS[b - 1]:<4} cloudiest year {worst:4.0%}   mean {mean:4.0%}   {now_txt}"

    print(f"  share of box with >= {MIN_VIEWS} clear views, best windows first:")
    for r in rows[:6]:
        print(line(r))
    mine = [r for r in rows if r[2] == current]
    if mine:
        print(f"  configured months:\n{line(mine[0])}")
