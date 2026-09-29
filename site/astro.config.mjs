// @ts-check
import { defineConfig } from 'astro/config';
import review from './integrations/review.mjs';

// https://astro.build/config
export default defineConfig({
  // Local-only /review page for model candidates; injected under `astro dev`, never built.
  integrations: [review()],
  // MapLibre loads its worker as a module worker.
  vite: { worker: { format: 'es' } },
});
