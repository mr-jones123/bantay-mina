const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');

/** Prefix a root-relative path with the deploy base (e.g. GitHub Pages project path). */
export function asset(path: string): string {
  return `${BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

/** Public source repository: pipeline, content and corrections (issues). */
export const REPO = 'https://github.com/mr-jones123/bantay-mina';
