# Bantay Mina

Before-and-after satellite images of mining sites across the Philippines, with the vegetation, tree
cover and shoreline change measured at each one. *Bantay* means "watch" or "guard".

The site shows what happened to the land, using images anyone can check. It is a static website. Every
image and number comes from one reproducible pipeline, and every data source is free and public.

## Sites

Seventeen sites: 10 in Luzon, 2 in the Visayas and 5 in Mindanao. "Green → bare" is the area that went
from dense vegetation to bare ground or water between the two images of the first comparison. Each page
also has a sharper Sentinel-2 comparison from 2016–18 to 2026.

| Island group | Site | Province | First comparison | Green → bare |
|---|---|---|---|---|
| Luzon | Barlo copper–gold–zinc mine (closed 1984) | Pangasinan | Four years after closure (1988–90) vs. 2026 | 9 ha |
| Luzon | Cagayan River mouth black sand and dredging | Cagayan | Before the river dredging (2020) vs. 2024 | 330 ha¹ |
| Luzon | Didipio gold–copper mine | Nueva Vizcaya | Before construction (2005–07) vs. 2026 | 172 ha |
| Luzon | Dinapigue nickel mine | Isabela | Before mine development (2003–06) vs. 2026 | 72 ha |
| Luzon | Itogon gold mining district | Benguet | Late large-mine era (1988–90) vs. 2026 | 29 ha |
| Luzon | Kasibu–Dupax exploration (baseline) | Nueva Vizcaya | Before exploration drilling (2024) vs. 2026 | 20 ha |
| Luzon | Lepanto gold–copper mine | Benguet | Operating mine, 1988–90 vs. 2026 | 16 ha |
| Luzon | Padcal copper–gold mine | Benguet | Operating mine, 1988–90 vs. 2026 | 143 ha |
| Luzon | Runruno gold mine | Nueva Vizcaya | Before mine construction (2006–09) vs. 2026 | 116 ha |
| Luzon | Santa Cruz–Candelaria nickel mines | Zambales | Before the nickel expansion (2002–05) vs. 2026 | 340 ha |
| Visayas | Semirara Island coal mines | Antique | Unong pit era (1988–90) vs. 2026 | 299 ha, plus ~2,370 ha of sea turned to land |
| Visayas | Toledo copper mine | Cebu | Earlier mining era (1992–95) vs. 2026 | 201 ha |
| Mindanao | Carrascal–Cantilan nickel mines | Surigao del Sur | Before large-scale nickel mining (1994–2002) vs. 2026 | 1,487 ha |
| Mindanao | Claver nickel mines and Taganito HPAL plant | Surigao del Norte | Before the nickel boom (1994–2002) vs. 2026 | 2,303 ha |
| Mindanao | Loreto nickel–chromite mines | Dinagat Islands | Before large-scale mining (1994–2002) vs. 2024–26 | 412 ha |
| Mindanao | Nonoc and Hinatuan Island nickel mines | Surigao del Norte | Between Nonoc's closure and restart (1994–2002) vs. 2026 | 419 ha |
| Mindanao | Tubajon–Libjo nickel mines | Dinagat Islands | Before large-scale mining (1994–2002) vs. 2025–26 | 220 ha |

¹ Mostly rice fields ploughed or harvested in one year and green in the other, not mining. The page says so.

## Ground rules

- **Facts, not accusations.** Images show land cover. They do not show pollution, health effects or
  whether an operation is legal. Those claims appear only in the text, each with a cited source.
  Allegations are attributed to the people who made them.
- **The build checks the sources.** A site page whose permit or timeline cites a URL that isn't in its
  source list fails the build.
- **Honest images.** Both sides of a comparison cover the same box on the same grid, use the same months
  of the year and the same colour stretch. Titles say when a "before" image is not before mining.
- **No people on the map.** No homes, villages, barricades or individuals are mapped. Indigenous
  communities are named only as public sources name them.
- **Permit boundaries are not on the site yet.** Mapped mine footprints show where land was dug, not
  where a company is allowed to dig. Until the Mines and Geosciences Bureau's tenement maps are added, the
  site doesn't say whether clearing happened inside or outside a permit.

## Data sources

| Data | Use | Licence |
|---|---|---|
| Landsat 5/7/8/9 Collection 2 Level-2 (USGS/NASA) | "before" images from 1988, "after" images | Public domain |
| Sentinel-2 Level-2A (ESA Copernicus) | 10 m images from 2016, and the cloud record | Free and open (Copernicus) |
| Hansen et al., Global Forest Change v1.13 | tree cover loss by year, land/water mask | CC BY 4.0 |
| Tang & Werner 2023, global mining footprint | mapped mine outlines | CC BY 4.0 |
| OpenStreetMap via OpenFreeMap | basemap | ODbL |

Landsat and Sentinel-2 are read through Microsoft Planetary Computer's public STAC API, with no API key.

## Repository layout

```
pipeline/   Python: builds composites, overlays and figures for each site
  sites.yaml     analysis boxes, months and before/after windows per site
  bantay/        grid, imagery (STAC), clouds, change (NDVI/NDWI), forest (Hansen), footprint, render
  bantay/ml/     experimental mining classifier (not used on the website)
site/       Astro: static site (MapLibre + OpenFreeMap basemap)
  src/content/sites/*.md    editorial content per site (sources required by schema)
  src/data/analysis/*.json  written by the pipeline, never edited by hand
  public/sites/<slug>/      images, overlays, GeoTIFFs written by the pipeline
```

## Run the pipeline

Needs [uv](https://docs.astral.sh/uv/).

One-off download of the Tang & Werner (2023) mine footprints (≈320 MB):

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
uv run python -m bantay clouds didipio  # which months give clear images here?
uv run python -m bantay build didipio   # or: --all
uv run python -m bantay retitle didipio # after editing only a comparison title in sites.yaml
```

A site takes about 3–15 minutes, depending on box size, and several sites can be built in parallel.
Most of the time goes on reading scenes from Planetary Computer.

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
   so, e.g. "Operating mine, 1988–90 vs. 2026". Landsat 7 is usable only before June 2003; after that
   its images have scan-line gaps. The Landsat archive over Caraga starts in 1994 and is thin, so those
   sites use every April–September view of 1994–2002, from Landsat 5 and 7 together.
2. Choose the months. Run `uv run python -m bantay clouds <slug>`. It measures the 2019–2025 Sentinel-2
   cloud record over the box and ranks runs of months by how much of the box had at least three clear
   views in the cloudiest year. Set `months` for the site if January–May isn't near the top. Luzon and
   Semirara: January–May. Caraga: April–September. Toledo: March–August.
3. Run `uv run python -m bantay build <slug>`, then open the images in `site/public/sites/<slug>/` and check
   them by eye. White cloud or grey no-data patches in an "after" image get counted as new bare ground.
   If you see them, add seasons to that side (e.g. `years: [2025, 2026]`) or allow cloudier scenes
   (`max_scene_cloud: 100`), and change the title to match.
4. Write `site/src/content/sites/<slug>.md` with `analysis: <slug>` and `islandGroup` (Luzon, Visayas or
   Mindanao). The build fails if any timeline entry or the permit cites a URL that is missing from
   `sources`.
5. Describe only what the images show. Claims about water, health or legality belong in the text, with a
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

## Mining classifier (experimental, not on the website)

A LightGBM model that looks at *open ground* (not dense green, not water) and scores how much it looks
like mine ground. It was meant to find new mining that no dataset maps yet. It is not part of the
website: in review it flagged too many villages, riverbeds and farm fields to be useful. The code stays
here for later work.

```sh
uv run python -m bantay.ml features            # cache 20 m feature stacks per region (~15 min, parallelisable)
uv run python -m bantay.ml train               # leave-one-region-out CV, final model, pipeline/ML_REPORT.md
uv run python -m bantay.ml predict --year 2026 # candidates for the Northern Luzon test regions
```

- **Regions** are defined in `pipeline/ml_regions.yaml`: 10 training regions outside Northern Luzon, and 5
  Northern Luzon test regions the model never sees during training.
- **Labels:** open ground inside Tang & Werner mine polygons counts as mining. Open ground more than 300 m
  from any mapped mine is a look-alike, labelled with its ESA WorldCover class.
- **Scores** are in `pipeline/ML_REPORT.md`. On Northern Luzon, average precision is 0.85. At the review
  threshold it finds about 93% of mapped mine ground, but only about 55% of what it flags is mine ground.
- **Review page (local only):** `cd site && npm run dev`, then open `http://localhost:4321/review`. It exists
  only under `astro dev` (through `integrations/review.mjs`); `npm run build` never includes it. Reviewers
  tick the visible signs of a mine and mark each candidate as mine ground, mining-affected, not mining or
  unsure. Candidates and decisions are saved under `pipeline/data/ml/`, which is not committed.

## Not done yet

- MGB mining tenement (permit) boundaries. Needed before the site can say whether any clearing happened
  outside a permit.
- NLMRC's Kasibu exploration permit: no public map or coordinates yet, so it has no image box. The
  Dupax (Woggle) box is placed from the company's own published target map.
- More sites: Palawan (Rio Tuba, Brooke's Point), Masbate (Aroroy), Marinduque (Marcopper), Davao de Oro,
  Zamboanga.
- The 99 ha cluster near Tuba, Benguet (120.566, 16.168) looks like a limestone quarry and cement
  plant; not verified and not included.

## Corrections and contact

If a date, figure or source is wrong, [open an issue](https://github.com/mr-jones123/bantay-mina/issues)
with a link to the evidence. Corrections are made openly and noted on the affected page.

Contact: [lacapxyniljhed@gmail.com](mailto:lacapxyniljhed@gmail.com) ·
[LinkedIn](https://www.linkedin.com/in/xy-lacap-76ba9029a/) · [GitHub](https://github.com/mr-jones123)
