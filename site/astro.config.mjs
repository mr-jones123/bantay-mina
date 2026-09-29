// @ts-check
import { defineConfig } from 'astro/config';
import review from './integrations/review.mjs';

// https://astro.build/config
export default defineConfig({
  // Deployed URL (Cloudflare Workers, see wrangler.jsonc); used for absolute share-image URLs.
  site: 'https://bantay-mina.xy-800.workers.dev',
  // Local-only /review page for model candidates; injected under `astro dev`, never built.
  integrations: [review()],
  // MapLibre loads its worker as a module worker.
  vite: { worker: { format: 'es' } },
});
