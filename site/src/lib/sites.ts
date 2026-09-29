import { getCollection, getEntry, type CollectionEntry } from 'astro:content';
import { asset } from './paths';

export interface MapSite {
  id: string;
  name: string;
  place: string;
  stage: string;
  href: string;
  center: [number, number];
  bounds: [number, number, number, number];
  footprint: string | null;
}

export interface LoadedSite {
  site: CollectionEntry<'sites'>;
  analysis: CollectionEntry<'analysis'>['data'];
}

export async function loadSites(): Promise<LoadedSite[]> {
  const sites = await getCollection('sites');
  const loaded = await Promise.all(
    sites.map(async (site) => {
      const entry = await getEntry(site.data.analysis);
      if (!entry) {
        throw new Error(`${site.id}: no analysis "${site.data.analysis.id}" — run the pipeline first`);
      }
      return { site, analysis: entry.data };
    }),
  );
  return loaded.sort((a, b) => a.site.data.name.localeCompare(b.site.data.name));
}

export function toMapSite({ site, analysis }: LoadedSite): MapSite {
  return {
    id: site.id,
    name: site.data.name,
    place: `${site.data.municipality}, ${site.data.province}`,
    stage: site.data.stage,
    href: asset(`/sites/${site.id}/`),
    center: analysis.center,
    bounds: analysis.bounds,
    footprint: analysis.footprint?.geojson ?? null,
  };
}

export const ha = (v: number) => `${Math.round(v).toLocaleString('en-US')} ha`;
