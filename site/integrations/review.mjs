/**
 * Local review page for the mining classifier. Exists only under `astro dev`: the route and its API
 * are injected in dev and never built, so unreviewed candidates cannot reach the published site.
 *
 * API (all under /__review/api, served by the Vite dev server):
 *   GET  /state                      regions, existing runs, model presence, current job
 *   GET  /run/:region/:year          meta + candidates with their review decisions merged in
 *   GET  /image/:region/:year/:file  before.webp | after.webp
 *   POST /review                     { region, year, candidateId, decision, reason?, note? }
 *   POST /jobs                       { region, year }  → features, then predict
 *   GET  /jobs/:id                   status + log tail
 */
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'yaml';

const PIPELINE = fileURLToPath(new URL('../../pipeline/', import.meta.url));
const ML = path.join(PIPELINE, 'data', 'ml');
const CANDIDATES = path.join(ML, 'candidates');
const REVIEWS = path.join(ML, 'reviews');
const PREFIX = '/__review/api';
// confirm = mine ground (pit, dump, pad, pond); affected = land changed by mining but not dug
// (downstream sediment or tailings); reject = something else (with a reason); unsure = look again.
const DECISIONS = new Set(['confirm', 'affected', 'reject', 'unsure']);
const REASONS = new Set(['riverbed', 'farmland', 'built-up', 'beach', 'landslide', 'quarry', 'road', 'cloud-or-shadow', 'other']);
// The review checklist: which signs of a mine the reviewer saw. Stored so decisions are comparable.
const SIGNS = new Set(['colour', 'engineered-shapes', 'haul-road', 'cut-into-slope', 'stays-bare', 'near-mining', 'muddy-water-downstream']);
const IMAGES = new Set(['before.webp', 'after.webp']);
const LOG_LINES = 400;

/** @returns {import('astro').AstroIntegration} */
export default function review() {
  return {
    name: 'bantay-review',
    hooks: {
      'astro:config:setup': ({ command, injectRoute, updateConfig, logger }) => {
        if (command !== 'dev') return;
        injectRoute({ pattern: '/review', entrypoint: new URL('../src/review/review.astro', import.meta.url) });
        updateConfig({ vite: { plugins: [reviewApi()] } });
        logger.info('local review page at /review (dev only)');
      },
    },
  };
}

function reviewApi() {
  return {
    name: 'bantay-review-api',
    /** @param {import('vite').ViteDevServer} server */
    configureServer(server) {
      server.middlewares.use(PREFIX, (req, res) => {
        handle(req, res).catch((err) => send(res, err.status ?? 500, { error: String(err.message ?? err) }));
      });
    },
  };
}

// ---------- helpers ----------

class HttpError extends Error {
  /** @param {number} status @param {string} message */
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/** @param {import('node:http').ServerResponse} res @param {number} status @param {unknown} body */
function send(res, status, body) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(body));
}

/** @param {import('node:http').IncomingMessage} req */
async function body(req) {
  let raw = '';
  for await (const chunk of req) raw += chunk;
  if (raw.length > 100_000) throw new HttpError(413, 'body too large');
  return raw ? JSON.parse(raw) : {};
}

async function regions() {
  const cfg = parse(await readFile(path.join(PIPELINE, 'ml_regions.yaml'), 'utf8'));
  /** @type {Record<string, {name: string, split: string, center: number[], boxKm: number}>} */
  const out = {};
  for (const split of ['test', 'train']) {
    for (const [name, r] of Object.entries(cfg[split] ?? {})) {
      out[name] = { name, split, center: r.center, boxKm: r.box_km };
    }
  }
  return out;
}

/** Validates region and year against the config, so nothing user-supplied reaches a path or a command. */
async function checkRun(region, year) {
  const all = await regions();
  const y = Number(year);
  if (!(region in all)) throw new HttpError(400, `unknown region ${region}`);
  if (!Number.isInteger(y) || y < 2019 || y > new Date().getFullYear()) throw new HttpError(400, `bad year ${year}`);
  return { region, year: y, dir: path.join(CANDIDATES, `${region}-${y}`) };
}

async function readJson(file, fallback) {
  return existsSync(file) ? JSON.parse(await readFile(file, 'utf8')) : fallback;
}

async function writeJsonAtomic(file, data) {
  await mkdir(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  await writeFile(tmp, JSON.stringify(data, null, 2));
  await rename(tmp, file);
}

const reviewsFile = (region) => path.join(REVIEWS, `${region}.json`);

async function listRuns() {
  if (!existsSync(CANDIDATES)) return [];
  const runs = [];
  for (const dir of await readdir(CANDIDATES)) {
    const meta = await readJson(path.join(CANDIDATES, dir, 'meta.json'), null);
    if (!meta) continue; // produced by an older predict without review support
    const fc = await readJson(path.join(CANDIDATES, dir, 'candidates.geojson'), { features: [] });
    const reviews = await readJson(reviewsFile(meta.region), []);
    const ids = new Set(fc.features.map((f) => f.properties.id));
    const linked = new Set(fc.features.map((f) => f.properties.reviewId).filter(Boolean));
    const reviewed = reviews.filter((r) => linked.has(r.reviewId) || (r.runId === meta.runId && ids.has(r.candidateId)));
    runs.push({ region: meta.region, year: meta.year, runId: meta.runId, generatedAt: meta.generatedAt,
                candidates: fc.features.length, reviewed: reviewed.length });
  }
  return runs.sort((a, b) => a.region.localeCompare(b.region) || b.year - a.year);
}

// ---------- jobs ----------

/** @type {Map<string, {id: string, region: string, year: number, status: string, log: string[], exitCode: number | null, startedAt: string}>} */
const jobs = new Map();

function currentJob() {
  return [...jobs.values()].find((j) => j.status === 'running') ?? null;
}

function startJob(region, year) {
  if (currentJob()) throw new HttpError(409, 'a run is already in progress');
  const job = { id: randomUUID(), region, year, status: 'running', log: [], exitCode: null,
                startedAt: new Date().toISOString() };
  jobs.set(job.id, job);
  const steps = [
    ['features', region, '--year', String(year)],
    ['predict', region, '--year', String(year)],
  ];
  const push = (text) => {
    for (const line of String(text).split(/\r?\n/)) {
      if (!line || line.includes('NotGeoreferencedWarning') || line.trim() === 'dest = _reproject(') continue;
      job.log.push(line);
    }
    if (job.log.length > LOG_LINES) job.log.splice(0, job.log.length - LOG_LINES);
  };
  const next = (i) => {
    if (i === steps.length) {
      job.status = 'done';
      return;
    }
    push(`$ uv run python -m bantay.ml ${steps[i].join(' ')}`);
    const child = spawn('uv', ['run', 'python', '-m', 'bantay.ml', ...steps[i]], { cwd: PIPELINE });
    child.stdout.on('data', push);
    child.stderr.on('data', push);
    child.on('error', (err) => {
      push(String(err));
      job.status = 'failed';
    });
    child.on('close', (code) => {
      job.exitCode = code;
      if (code === 0) next(i + 1);
      else job.status = 'failed';
    });
  };
  next(0);
  return job;
}

// ---------- routes ----------

/** @param {import('node:http').IncomingMessage} req @param {import('node:http').ServerResponse} res */
async function handle(req, res) {
  const url = new URL(req.url ?? '/', 'http://local');
  const parts = url.pathname.split('/').filter(Boolean);

  if (req.method === 'GET' && parts[0] === 'state') {
    return send(res, 200, {
      regions: Object.values(await regions()),
      runs: await listRuns(),
      modelTrained: existsSync(path.join(ML, 'model', 'model.joblib')),
      job: currentJob(),
    });
  }

  if (req.method === 'GET' && parts[0] === 'run' && parts.length === 3) {
    const { region, dir } = await checkRun(parts[1], parts[2]);
    const meta = await readJson(path.join(dir, 'meta.json'), null);
    if (!meta) throw new HttpError(404, 'no run for that region and year yet');
    const fc = await readJson(path.join(dir, 'candidates.geojson'), { type: 'FeatureCollection', features: [] });
    const reviews = await readJson(reviewsFile(region), []);
    const byId = new Map(reviews.map((r) => [r.reviewId, r]));
    const byCandidate = new Map(reviews.filter((r) => r.runId === meta.runId).map((r) => [r.candidateId, r]));
    for (const f of fc.features) {
      f.properties.review = byId.get(f.properties.reviewId) ?? byCandidate.get(f.properties.id) ?? null;
    }
    return send(res, 200, { meta, candidates: fc });
  }

  if (req.method === 'GET' && parts[0] === 'image' && parts.length === 4) {
    const { dir } = await checkRun(parts[1], parts[2]);
    if (!IMAGES.has(parts[3])) throw new HttpError(400, 'unknown image');
    const file = path.join(dir, parts[3]);
    if (!existsSync(file)) throw new HttpError(404, 'image missing; re-run predict');
    res.setHeader('Content-Type', 'image/webp');
    res.setHeader('Cache-Control', 'no-store');
    return res.end(await readFile(file));
  }

  if (req.method === 'POST' && parts[0] === 'review') {
    const b = await body(req);
    const { region, year, dir } = await checkRun(b.region, b.year);
    const meta = await readJson(path.join(dir, 'meta.json'), null);
    const fc = await readJson(path.join(dir, 'candidates.geojson'), { features: [] });
    const feature = fc.features.find((f) => f.properties.id === b.candidateId);
    if (!meta || !feature) throw new HttpError(404, 'candidate not found');
    if (b.decision !== null && !DECISIONS.has(b.decision)) throw new HttpError(400, 'bad decision');
    if (b.reason && !REASONS.has(b.reason)) throw new HttpError(400, 'bad reason');

    const file = reviewsFile(region);
    const reviews = await readJson(file, []);
    const existingId = feature.properties.reviewId;
    const idx = reviews.findIndex((r) => r.reviewId === existingId || (r.runId === meta.runId && r.candidateId === b.candidateId));
    if (b.decision === null) {
      if (idx >= 0) reviews.splice(idx, 1);
      await writeJsonAtomic(file, reviews);
      return send(res, 200, { review: null });
    }
    const p = feature.properties;
    const record = {
      reviewId: idx >= 0 ? reviews[idx].reviewId : randomUUID(),
      runId: meta.runId,
      candidateId: b.candidateId,
      year,
      decision: b.decision,
      reason: b.decision === 'reject' ? (b.reason ?? 'other') : null,
      note: String(b.note ?? '').slice(0, 2000),
      signs: Array.isArray(b.signs) ? b.signs.filter((s) => SIGNS.has(s)) : [],
      reviewedAt: new Date().toISOString(),
      areaHa: p.areaHa,
      meanProbability: p.meanProbability,
      lon: p.lon,
      lat: p.lat,
      geometry: feature.geometry, // decisions become training labels, so keep the outline
    };
    if (idx >= 0) reviews[idx] = record;
    else reviews.push(record);
    await writeJsonAtomic(file, reviews);
    return send(res, 200, { review: record });
  }

  if (req.method === 'POST' && parts[0] === 'jobs') {
    const b = await body(req);
    const { region, year } = await checkRun(b.region, b.year);
    if (!existsSync(path.join(ML, 'model', 'model.joblib'))) {
      throw new HttpError(409, 'no trained model; run `uv run python -m bantay.ml train` first');
    }
    return send(res, 202, startJob(region, year));
  }

  if (req.method === 'GET' && parts[0] === 'jobs' && parts.length === 2) {
    const job = jobs.get(parts[1]);
    if (!job) throw new HttpError(404, 'unknown job');
    return send(res, 200, job);
  }

  throw new HttpError(404, 'not found');
}
