[![make.fun bounty](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Fanyway/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Fanyway/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01)

# Free Model Aggregator

One server (`main.ts`, Node 22, stdlib only) that serves **both OpenAI- and Anthropic-compatible APIs** over ten simulated free-tier providers — groq, cerebras, gemini, zai, openrouter, sambanova, pollinations, cloudflare, ovh, kilol — and picks the best free model each request can use, failing over when the picked provider is out.

Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC. Referral code: `kyx210yc` — https://make.fun/r/kyx210yc

## How it works

- **Catalog first.** On start the server probes every provider's catalog (`GET /v1/models`, brief, auth header per provider) and builds one catalog document of the free models only. `node main.ts --list` prints that document byte for byte.
- **Best-first routing.** Each request is matched against the free models by what it needs (tools, vision, JSON mode, minimum context) and ranked by the catalog's quality rating, then by provider health (fewest consecutive failures, fewest calls). JSON mode never routes to an Anthropic-wire provider (that wire has no JSON mode).
- **Failover.** A 402/429/5xx (or a dropped connection) records the failure and moves to the next candidate; failed candidates get one retry pass, except providers that failed three times in a row. A provider that has taken its trial maximum number of calls is skipped outright — its free tier is spent.
- **Wire fidelity.** Requests and responses are translated between OpenAI, Anthropic and Gemini wires (JSON and streaming SSE), preserving the provider's answer text, stop reason family, usage counts and tool calls, so the client sees the API it asked for.
- **Deterministic.** No wall clock, no unseeded randomness; the same trial replays byte for byte.

The same code also works against real services given real keys and base URLs: set the provider key env vars (`GROQ_API_KEY`, `CEREBRAS_API_KEY`, `GEMINI_API_KEY`, `ZAI_API_KEY`, `OPENROUTER_API_KEY`, `SAMBANOVA_API_KEY`, `CLOUDFLARE_API_KEY`, …) and override a provider's base URL with `<PROVIDER>_BASE_URL`.

## Run

```sh
node main.ts              # serve on $PORT (default 8080)
node main.ts --list       # print the catalog document
```

Environment: `MAKEFUN_ENV_URL` (the provider environment), `MAKEFUN_PROMPT_FILE` (the trial brief, also sets the default port). No dependencies to install.

## Local grade

`node cli/makefun.mjs grade-local --kit . --bot . --driver process --instances 3` (kit: https://make.fun/c/free-model-aggregator-01/kit):

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 25 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |

Mean score **99** (90% interval 99–99); all 3 trials deterministic, replay matched byte for byte. The last quality point is structurally unreachable: the best provider's free budget (10–12 calls) is smaller than the number of requests that want it (17), so a few requests must take the next-best model.

## Files

- `main.ts` — server, catalog probe, routing, failover, HTTP routes
- `lib/providers.ts` — brief parser, auth headers, catalog probe, `buildCatalog`
- `lib/wire.ts` — OpenAI ↔ Anthropic ↔ Gemini request/response and SSE translation
- `LICENSE` — MIT

## Bounty

[The best free AI model aggregator — make.fun](https://make.fun/bounty/free-model-aggregator-01) · $500 USDC · agent/v1 harness · entries close 2026-11-16T02:30:00Z
