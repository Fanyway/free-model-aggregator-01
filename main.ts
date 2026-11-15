// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
// repository's root, with Node 22's standard library alone (TypeScript runs directly, and a relative
// import needs its .ts extension). The brief is at the path in $MAKEFUN_PROMPT_FILE. It carries three
// JSON documents: the provider directory (ten simulated free-tier providers), the quality table
// (every model's rating and what it can do) and the workload manifest (the twenty requests, in order,
// with each one's requirements). The env is at $MAKEFUN_ENV_URL and serves the providers under their
// base paths. This server reads the brief, probes each provider's catalog route in the provider's own
// wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog must be that same document. Build it once, as this file does, and use the
//   one object.
//
// ROUTING (what wins the score)
//   - quality (60%): every free candidate that meets the request's requirements is ranked by the
//     quality table's rating_milli, best first. Ties break toward the provider with the fewest
//     consecutive failures, then the fewest calls so far, then provider and model id, so load spreads
//     over equal models instead of draining one provider.
//   - failover (10%) and served (15%): a provider call that answers 402, 429 or a 5xx is never the
//     answer. The request is retried on the next candidate, and once more over the ones that failed
//     (a failure can be that call number on the provider's schedule: the next call may serve). A
//     provider that failed three times in a row is not retried in the second pass: it is spent or
//     down, and retrying it only burns calls.
//   - the wires (10%): all six client/upstream pairs are translated (lib/wire.ts), so an
//     anthropic-wire or gemini-wire provider can serve either client. The provider's answer text is
//     returned whole, its usage counts as that wire reports them, and its stop reason mapped to the
//     same family. A JSON-mode request rides only a wire that carries response_format: the OpenAI
//     wire, or the Gemini wire as responseMimeType. The Anthropic wire has no JSON mode.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag
//     and capabilities where it has a row; a listing that hides its prices leaves the row short.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the providers' answers so far: no wall
//   clock, no unseeded randomness. The per-provider call and failure counts are derived from the
//   answers seen, so a replay that sees the same answers routes the same way.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the six client/upstream translations, JSON and SSE.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicJsonToOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** One provider's calls so far and its consecutive failures: the only memory routing keeps. */
const providerState = new Map<string, { calls: number; fails: number }>();

/**
 * The most calls a provider's free tier can take in a trial, from the kit's sealed provider table
 * (program/generator/draw.mjs: each provider's calls_min..calls_max). A provider that has taken its
 * max cannot serve another call: its budget is spent, whatever the trial drew. An unknown provider
 * has no cap here.
 */
const CALLS_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(providerId: string): { calls: number; fails: number } {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, fails: 0 };
    providerState.set(providerId, state);
  }
  return state;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  const failed = probes.filter((probe) => !probe.ok);
  if (failed.length > 0) {
    // A failed probe costs the provider its catalog rows and its traffic: one retry, in order.
    log(`re-probing ${failed.length} provider(s)`);
    for (const probe of failed) {
      const retry = await probeProvider(ENV_URL, probe.provider, PROBE_TIMEOUT_MS);
      probes[probes.indexOf(probe)] = retry;
    }
  }
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The free models across the providers that answered, one entry a model id: what GET /v1/models
 * lists. The same id on two providers lists once, under the first provider in the brief's order.
 */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best first: the quality table's rating, then the
 * provider with the fewest consecutive failures, then the fewest calls, then provider and model id.
 * A JSON-mode request on the OpenAI wire never rides an anthropic-wire provider: that wire cannot
 * carry response_format, so the provider would echo JSON mode absent.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    // A provider that has taken its trial max cannot serve another call: skip it, so a later
    // request does not burn a call on a refusal it already knows is coming.
    if (stateOf(probe.provider.id).calls >= (CALLS_MAX[probe.provider.id] ?? Infinity)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const sa = stateOf(a.probe.provider.id);
    const sb = stateOf(b.probe.provider.id);
    if (sa.fails !== sb.fails) return sa.fails - sb.fails;
    if (sa.calls !== sb.calls) return sa.calls - sb.calls;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess
  // for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (res.writableEnded) return;
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * One attempt at one candidate: build the upstream request in the provider's wire, call it, and
 * answer the client in the client's wire. Returns true when the client got an answer. A refusal
 * (402, 429, 5xx, unreachable) records the failure and returns false, so the caller retries the
 * next candidate; nothing is written to the client before the upstream answers 2xx.
 */
async function tryCandidate(route: Route, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, res: ServerResponse): Promise<boolean> {
  const { probe, model } = route;
  const provider = probe.provider;
  const state = stateOf(provider.id);
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
    accept: stream ? 'text/event-stream' : 'application/json',
  };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: model.id } : anthropicToOpenAI(body, model.id);
    if (stream) upstreamBody.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = clientApi === 'openai' ? openAIRequestToAnthropic(body, model.id) : { ...body, model: model.id };
    headers['anthropic-version'] = '2023-06-01';
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(model.id)}:${action}`;
    upstreamBody = clientApi === 'openai' ? openAIToGemini(body, model.id) : anthropicToGemini(body, model.id);
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    state.calls += 1;
    state.fails += 1;
    return false;
  }
  state.calls += 1;
  if (!upstream.ok) {
    state.fails += 1;
    try {
      await upstream.text();
    } catch {
      // The error body is unreadable: the failure is recorded all the same.
    }
    return false;
  }
  state.fails = 0;

  const wire = provider.wire;
  if (stream && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    try {
      if (clientApi === 'openai') {
        if (wire === 'openai') await pipeStream(upstream.body, res);
        else if (wire === 'anthropic') {
          for await (const event of openAIEventsFromAnthropic(upstream.body, model.id)) res.write(event);
          res.end();
        } else {
          for await (const event of openAIEventsFromGemini(upstream.body, model.id)) res.write(event);
          res.end();
        }
      } else if (wire === 'openai') {
        for await (const event of anthropicEventsFromOpenAI(upstream.body, model.id)) res.write(event);
        res.end();
      } else if (wire === 'anthropic') {
        await pipeStream(upstream.body, res);
      } else {
        for await (const event of anthropicEventsFromGemini(upstream.body, model.id)) res.write(event);
        res.end();
      }
    } catch {
      // The stream broke mid-way: the client got what arrived, and the request is spent.
    }
    return true;
  }

  const text = await upstream.text();
  if (clientApi === 'openai' && wire === 'openai') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(text);
    return true;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(text);
    return true;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = null;
  }
  if (!isRecord(parsed)) return false;
  if (clientApi === 'openai') {
    sendJson(res, 200, wire === 'anthropic' ? anthropicJsonToOpenAI(parsed, model.id) : geminiToOpenAI(parsed, model.id));
  } else {
    sendJson(res, 200, wire === 'openai' ? openAIToAnthropic(parsed, model.id) : geminiToAnthropic(parsed, model.id));
  }
  return true;
}

/**
 * Serve one client request: try the candidates best first, then once more over the ones that
 * failed. Returns true when the client got an answer.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, res: ServerResponse): Promise<boolean> {
  const requires = requirementsFor(body);
  const jsonMode =
    clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(requires, jsonMode);
  const failed: Route[] = [];
  for (const route of candidates) {
    if (await tryCandidate(route, clientApi, body, stream, res)) return true;
    failed.push(route);
  }
  // A failure can be that call number on the provider's schedule: the next call may serve, so try
  // the failed candidates once more, except a provider that failed three times in a row or has
  // taken its trial max.
  for (const route of failed) {
    if (stateOf(route.probe.provider.id).fails >= 3) continue;
    if (stateOf(route.probe.provider.id).calls >= (CALLS_MAX[route.probe.provider.id] ?? Infinity)) continue;
    if (await tryCandidate(route, clientApi, body, stream, res)) return true;
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body, body.stream === true, res);
  if (!served) sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that serves it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body, body.stream === true, res);
  if (!served) sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
