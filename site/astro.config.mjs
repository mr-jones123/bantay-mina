// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  // MapLibre loads its worker as a module worker.
  vite: { worker: { format: 'es' } },
});
