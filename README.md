# Bantay Mina

Before-and-after satellite images of mining sites in Northern Luzon, with the vegetation and tree cover
lost at each one. It is a static website. The images and numbers come from one reproducible pipeline,
and every data source is free and public.

```
pipeline/   Python: builds composites, overlays and figures for each site
  sites.yaml     analysis boxes and before/after windows per site
  bantay/        grid, imagery (STAC), change (NDVI), forest (Hansen), footprint (Tang & Werner), render
site/       Astro: static site (MapLibre + OpenFreeMap basemap)
  src/content/sites/*.md    editorial content per site (sources required by schema)
  src/data/analysis/*.json  written by the pipeline, never edited by hand
  public/sites/<slug>/      images, overlays, GeoTIFFs written by the pipeline
```

## Run the pipeline

Needs [uv](https://docs.astral.sh/uv/). No API keys: imagery comes from Microsoft Planetary Computer's
public STAC API.

One-off download of the Tang & Werner (2023) mine footprints (≈320 MB, CC BY 4.0):

```sh
cd pipeline
mkdir -p data/raw/tang_werner && cd data/raw/tang_werner
for ext in dbf cpg shp shx prj; do
  curl -L -o "tw.$ext" "https://zenodo.org/api/records/7894216/files/74548_projected%20polygons.$ext/content"
done
cd ../../..
```

Then build a site. The first run also downloads the Hansen tree cover, loss and land/water tiles
(≈210 MB per 10° tile):

```sh
uv run python -m bantay build didipio   # or: --all
uv run python -m bantay retitle didipio # after editing only a comparison title in sites.yaml
```

A site takes about 5–10 minutes, and several sites can be built in parallel. Most of the time goes on
reading scenes from Planetary Computer.

Current sites: `didipio`, `padcal`, `lepanto`, `runruno`, `dinapigue`, `zambales-nickel`, `itogon`, `barlo`,
`cagayan-black-sand`, `kasibu-dupax-exploration`.

## Run the site

Needs Node ≥ 22.12.

```sh
cd site
npm install
npm run dev        # http://localhost:4321
npm run build      # static output in site/dist/
```

`site/dist/` can be hosted on any static host (Cloudflare Pages, GitHub Pages, Netlify). For a GitHub
Pages project path, set `base` in `site/astro.config.mjs`. All links go through `asset()`, so they pick up
the base automatically.

## Add a site

1. Add an entry to `pipeline/sites.yaml`: the centre point, plus before/after years for each comparison.
   The Tang & Werner footprint clusters are a good way to find centres. Keep "before" earlier than the
   first clearing where imagery allows it; if it can't be (for mines older than Landsat), the title must say
   so, e.g. "Operating mine, 1988–90 vs. 2026". Use only Landsat 5 for pre-2012 dates (Landsat 7 has
   scan-line gaps).
2. Run `uv run python -m bantay build <slug>`, then open the images in `site/public/sites/<slug>/` and check
   them by eye.
3. Write `site/src/content/sites/<slug>.md` with `analysis: <slug>`. The build fails if any timeline entry
   or the permit cites a URL that is missing from `sources`.
4. Describe only what the images show. Claims about water, health or legality belong in the text, with a
   source.

## What the numbers mean

- **Composites**: for each pixel, the clear view (after cloud masking) that ranks a quarter of the way up
  from the darkest in blue. Missed cloud is bright, so this rejects it. All bands come from that one view.
- **Vegetation → bare**: land pixels with NDVI ≥ 0.5 in the "before" composite and < 0.3 in the "after"
  composite. Counts any clearing (farms and brush as well as forest), so the pipeline also reports the part
  inside the mapped mine footprint. Permanent water (Hansen data mask) is excluded; new ponds are counted.
- **Water change**: NDWI < −0.1 on one side and > 0.1 on the other (land → water, water → land). Inside
  mapped permanent water, very dark pixels count as water; only former land can become new water.
- **Tree cover loss**: Hansen GFC-2025-v1.13, ≥ 30% canopy in 2000, all causes. It counts only trees and
  never subtracts regrowth.
- **Mine footprint**: Tang & Werner 2023 polygons, clipped to the analysis box. These are not permit
  boundaries.

The full explanation is on the site's Methods page.

## Mining classifier (candidate finder)

A LightGBM model that looks at *open ground* (not dense green, not water) and scores how much it looks
like mine ground. It produces a review list, never a published finding.

```sh
uv run python -m bantay.ml features            # cache 20 m feature stacks per region (~15 min, parallelisable)
uv run python -m bantay.ml train               # leave-one-region-out CV, final model, pipeline/ML_REPORT.md
uv run python -m bantay.ml predict --year 2026 # candidates for the Northern Luzon regions
```

- **Regions** are defined in `pipeline/ml_regions.yaml`. The model trains on 10 regions outside Northern
  Luzon (Mindanao and Dinagat nickel, Cebu, Palawan, Masbate, Semirara, Marinduque, Negros, southern
  Zambales). It is scored once on 5 Northern Luzon regions it never saw during training.
- **Labels:** open ground inside Tang & Werner mine polygons counts as mining. Open ground more than 300 m
  from any mapped mine is a look-alike, labelled with its ESA WorldCover class (cropland, built-up,
  bare/sparse, grass/shrub, other).
- **Features** (`bantay/ml/features.py`), all measured in the dry season:
  - Sentinel-2 bands including the shortwave-infrared B11 and B12
  - iron-oxide, clay and bare-soil indices
  - the same measures three years earlier, and the change since
  - share of bare ground within 220 m and 500 m, and local texture
  - Sentinel-1 radar backscatter
  - slope, and terrain position relative to the surrounding 500 m
  - JRC surface-water occurrence since 1984
  - Hansen tree cover in 2000, and years since tree loss
  - Absolute elevation is left out on purpose: the first model used it as a shortcut.
- **Scores** are in `pipeline/ML_REPORT.md`. On Northern Luzon, average precision is 0.85. At the review
  threshold, it finds about 93% of mapped mine ground, but only about 55% of what it flags is mine ground.
  So the list needs a person to check it.
- **Known false alarms:**
  - riverbeds and sediment in narrow mountain valleys
  - dry-season farm plots on slopes
  - dense built-up areas (Baguio)
  - beaches
- **Candidates** are written to `pipeline/data/ml/candidates/<region>-<year>/`. Each run produces
  `candidates.geojson`, `candidates.csv` and a `review.png` contact sheet. They are not committed and not
  on the website. Each candidate starts as `review: unreviewed`.

## Not done yet

- MGB mining tenement (permit) boundaries. Needed before the site can say whether any clearing happened
  outside a permit.
- NLMRC's Kasibu exploration permit: no public map or coordinates yet, so it has no image box. The
  Dupax (Woggle) box is placed from the company's own published target map.
- The 99 ha cluster near Tuba, Benguet (120.566, 16.168) looks like a limestone quarry and cement
  plant; not verified and not included.
- Reviewing the candidate lists. Confirmed and rejected candidates should become new labels, especially
  the rejected ones, so the next model learns those look-alikes.
