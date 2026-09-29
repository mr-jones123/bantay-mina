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

## Not done yet

- MGB mining tenement (permit) boundaries. Needed before the site can say whether any clearing happened
  outside a permit.
- NLMRC's Kasibu exploration permit: no public map or coordinates yet, so it has no image box. The
  Dupax (Woggle) box is placed from the company's own published target map.
- The 99 ha cluster near Tuba, Benguet (120.566, 16.168) looks like a limestone quarry and cement
  plant; not verified and not included.
- The ML phase: first detecting new bare land, then classifying which of it is mining.
