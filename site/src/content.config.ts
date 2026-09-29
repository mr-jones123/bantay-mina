import { defineCollection, reference } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

// Written by the pipeline (`pipeline/bantay/__main__.py`); never edit by hand.
const side = z.object({
  years: z.array(z.number().int()),
  months: z.tuple([z.number().int(), z.number().int()]),
  platforms: z.array(z.string()),
  platformLabel: z.string(),
  sceneCount: z.number().int().positive(),
  sceneDates: z.array(z.string()),
  sceneIds: z.array(z.string()),
  clearFraction: z.number(),
  medianClearObservations: z.number(),
  image: z.string(),
  geotiff: z.string(),
});

const lossStats = z.object({
  areaHa: z.number(),
  treeCover2000Ha: z.number(),
  lossHa: z.number(),
  lossByYear: z.record(z.string(), z.number()),
});

const analysis = defineCollection({
  loader: glob({ base: './src/data/analysis', pattern: '*.json' }),
  schema: z.object({
    slug: z.string(),
    generatedAt: z.string(),
    center: z.tuple([z.number(), z.number()]),
    boxKm: z.number(),
    bounds: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    crs: z.string(),
    pixelM: z.number(),
    sizePx: z.tuple([z.number().int(), z.number().int()]),
    stretch: z.tuple([z.number(), z.number()]),
    gamma: z.number(),
    comparisons: z
      .array(
        z.object({
          id: z.string(),
          title: z.string(),
          sensor: z.string(),
          nativeResolutionM: z.number(),
          attribution: z.string(),
          before: side,
          after: side,
          change: z.object({
            ndviVegetated: z.number(),
            ndviBare: z.number(),
            comparableFraction: z.number(),
            vegetatedBeforeHa: z.number(),
            vegetationToBareHa: z.number(),
            insideFootprintHa: z.number().nullable(),
            overlay: z.string(),
          }),
          water: z.object({
            ndwiWater: z.number(),
            ndwiDry: z.number(),
            landToWaterHa: z.number(),
            waterToLandHa: z.number(),
            landToWaterInsideFootprintHa: z.number().nullable(),
            overlay: z.string(),
          }),
        }),
      )
      .min(1),
    footprint: z
      .object({
        source: z.string(),
        citation: z.string(),
        url: z.url(),
        polygons: z.number().int(),
        areaHa: z.number(),
        svgPath: z.string(),
        geojson: z.string(),
      })
      .nullable(),
    forest: z.object({
      dataset: z.string(),
      citation: z.string(),
      canopyThresholdPct: z.number(),
      years: z.tuple([z.number().int(), z.number().int()]),
      box: lossStats,
      footprint: lossStats.nullable(),
      overlay: z.string(),
    }),
  }),
});

// Editorial content. The schema is the fact-checking gate: a case without
// sources, or a timeline claim without a listed source, fails the build.
// YAML turns unquoted 2021-07-14 into a Date; normalise back to the string form.
const partialDate = z.preprocess(
  (v) => (v instanceof Date ? v.toISOString().slice(0, 10) : v),
  z.string().regex(/^\d{4}(-\d{2}(-\d{2})?)?$/, 'use YYYY, YYYY-MM or YYYY-MM-DD'),
);

const source = z.object({
  title: z.string(),
  publisher: z.string(),
  url: z.url(),
  date: partialDate.optional(),
});

const sites = defineCollection({
  loader: glob({ base: './src/content/sites', pattern: '*.md' }),
  schema: z
    .object({
      name: z.string(),
      analysis: reference('analysis'),
      municipality: z.string(),
      province: z.string(),
      region: z.string(),
      operator: z.string(),
      commodities: z.array(z.string()).min(1),
      stage: z.enum(['exploration', 'development', 'operating', 'suspended', 'closed']),
      summary: z.string().max(320),
      permit: z.object({
        type: z.string(),
        holder: z.string(),
        status: z.string(),
        source: z.url(),
      }),
      timeline: z
        .array(
          z.object({
            date: partialDate,
            text: z.string(),
            sources: z.array(z.url()).min(1),
          }),
        )
        .min(1),
      sources: z.array(source).min(1),
      lastReviewed: z.coerce.date(),
    })
    .superRefine((site, ctx) => {
      const listed = new Set(site.sources.map((s) => s.url));
      const cited = [site.permit.source, ...site.timeline.flatMap((t) => t.sources)];
      for (const url of cited) {
        if (!listed.has(url)) {
          ctx.addIssue({ code: 'custom', message: `cited source not in sources list: ${url}` });
        }
      }
    }),
});

export const collections = { analysis, sites };
