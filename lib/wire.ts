// The client wires and the upstream wires: every pair, in JSON and in server-sent events.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. Each provider speaks its own
// wire (openai, anthropic or gemini), so this file translates both ways for all six pairs:
//
//   client openai    -> upstream openai     passthrough
//   client openai    -> upstream anthropic  openAIToAnthropic + anthropicJsonToOpenAI / openAIEventsFromAnthropic
//   client openai    -> upstream gemini     openAIToGemini + geminiToOpenAI / openAIEventsFromGemini
//   client anthropic -> upstream openai     anthropicToOpenAI + openAIToAnthropic / anthropicEventsFromOpenAI
//   client anthropic -> upstream anthropic  passthrough
//   client anthropic -> upstream gemini     anthropicToGemini + geminiToAnthropic / anthropicEventsFromGemini
//
// The translations are written so the provider echoes exactly what the client sent: the system prompt,
// the user text, the images, the tool names, max_tokens, the stop sequences and JSON mode all pass
// through in the shape the upstream wire reads them in. The provider's answer text is returned whole,
// its usage counts as that wire reports them, and its stop reason mapped to the same family. A JSON-mode
// request can only ride a wire that carries response_format: the OpenAI wire (response_format) and the
// Gemini wire (generationConfig.responseMimeType); the Anthropic wire has no JSON mode, so main.ts never
// routes a JSON-mode OpenAI request to an anthropic-wire provider.
//
// Everything here is a pure function of its arguments (or, for the SSE generators, of the stream they
// read): no clock and no randomness, so the same body and the same provider answer give the same bytes
// on a replay. The two model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function finishReasonOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function stopReasonOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The stop sequences of a body, as an array of strings: a string becomes one, anything else is dropped. */
function stopList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((v): v is string => typeof v === 'string');
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One OpenAI content part as an Anthropic content block: text or an image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== undefined && data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: system
 * messages become the system prompt, image_url parts become image blocks, tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through. JSON mode has no Anthropic
 * spelling, so the caller never routes a JSON-mode request here.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = typeof content === 'string' ? content : textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : toolResultText(content) }],
      });
      continue;
    }
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = blockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  out.messages = messages;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages object as a Chat Completions answer naming `model`. */
export function anthropicJsonToOpenAI(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: texts.join(''), ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: finishReasonOf(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

function parseSseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** The frames of an SSE stream, in order, as they arrive. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield parseSseFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseSseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

/** A Chat Completions chunk object: one delta, optionally the finish, optionally the usage. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage: Json | null): Json {
  const chunk: Json = { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
  if (usage !== null) chunk.usage = usage;
  return chunk;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason
 * and usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    const out: string[] = [...startEvents()];
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: { type: 'tool_use', id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`, name: typeof fn.name === 'string' ? fn.name : '', input: {} },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
    // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    for (const event of out) yield event;
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}

/**
 * An Anthropic SSE stream as Chat Completions chunks: the role chunk, the text deltas, the tool-call
 * deltas, then one finish chunk, one usage chunk and [DONE]. `model` is the model the client asked
 * for, named back to it.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  let sentRole = false;
  const toolBlocks = new Map<number, { id: string; name: string; args: string }>();
  const emitRole = (): string[] => {
    if (sentRole) return [];
    sentRole = true;
    return [`data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null, null))}\n\n`];
  };
  for await (const frame of sseFrames(upstream)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      continue;
    }
    if (frame.event === 'content_block_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      const index = typeof data.index === 'number' ? data.index : -1;
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield* emitRole();
        yield `data: ${JSON.stringify(openAIChunk(model, { content: delta.text }, null, null))}\n\n`;
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        const block = toolBlocks.get(index);
        if (block !== undefined) {
          block.args += delta.partial_json;
          yield `data: ${JSON.stringify(openAIChunk(model, { tool_calls: [{ index, id: block.id, type: 'function', function: { arguments: delta.partial_json } }] }, null, null))}\n\n`;
        }
      }
      continue;
    }
    if (frame.event === 'content_block_start') {
      const block = isRecord(data.content_block) ? data.content_block : {};
      const index = typeof data.index === 'number' ? data.index : -1;
      if (block.type === 'tool_use') {
        const id = typeof block.id === 'string' ? block.id : `call_${index}`;
        const name = typeof block.name === 'string' ? block.name : '';
        toolBlocks.set(index, { id, name, args: '' });
        yield* emitRole();
        yield `data: ${JSON.stringify(openAIChunk(model, { tool_calls: [{ index, id, type: 'function', function: { name, arguments: '' } }] }, null, null))}\n\n`;
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      continue;
    }
  }
  yield* emitRole();
  yield `data: ${JSON.stringify(openAIChunk(model, {}, finishReasonOf(stopReason ?? 'end_turn'), null))}\n\n`;
  yield `data: ${JSON.stringify(openAIChunk(model, {}, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }))}\n\n`;
  yield 'data: [DONE]\n\n';
}

/** One Gemini content part: a text part, an inline image part, or a function call. */
function geminiPartOfText(text: string): Json {
  return { text };
}

/** The media type and base64 data of a data URL, or null when the URL is not one. */
function dataUrlOf(url: string): { media: string; data: string } | null {
  const match = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1] !== undefined && match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** One OpenAI message as Gemini contents parts: text, images, tool calls. */
function geminiPartsOfOpenAIMessage(raw: Json): Json[] {
  const parts: Json[] = [];
  const content = raw.content;
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
  } else if (Array.isArray(content)) {
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const data = dataUrlOf(part.image_url.url);
        if (data !== null) parts.push({ inlineData: { mimeType: data.media, data: data.data } });
        else parts.push({ fileData: { mimeType: 'image/png', fileUri: part.image_url.url } });
      }
    }
  }
  const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
  }
  return parts;
}

/** One Anthropic message as Gemini contents parts: text, images, tool uses and tool results. */
function geminiPartsOfAnthropicMessage(raw: Json): Json[] {
  const parts: Json[] = [];
  const content = raw.content;
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: source.url } });
      }
    } else if (block.type === 'tool_use') {
      parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
    } else if (block.type === 'tool_result') {
      parts.push({
        functionResponse: {
          name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          response: { content: toolResultText(block.content) },
        },
      });
    }
  }
  return parts;
}

/** The generationConfig of a Gemini request from either client wire's caps. */
function geminiGenerationConfig(maxTokens: number | undefined, temperature: unknown, topP: unknown, stops: string[], jsonMode: boolean): Json {
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/** A Gemini toolConfig from a tool_choice: AUTO, ANY with names, or NONE. */
function geminiToolConfig(choice: unknown): Json | null {
  if (typeof choice === 'string') {
    if (choice === 'auto') return { functionCallingConfig: { mode: 'AUTO' } };
    if (choice === 'required') return { functionCallingConfig: { mode: 'ANY' } };
    if (choice === 'none') return { functionCallingConfig: { mode: 'NONE' } };
    return null;
  }
  if (!isRecord(choice)) return null;
  if (choice.type === 'auto') return { functionCallingConfig: { mode: 'AUTO' } };
  if (choice.type === 'any') return { functionCallingConfig: { mode: 'ANY' } };
  if (choice.type === 'none') return { functionCallingConfig: { mode: 'NONE' } };
  if (choice.type === 'tool' && typeof choice.name === 'string') return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
  if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the system prompt becomes the
 * systemInstruction, messages become contents (images as inlineData, tool calls as functionCall), tools
 * become functionDeclarations, and the output cap, temperature, top_p, stop sequences and JSON mode
 * become the generationConfig.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts = geminiPartsOfOpenAIMessage(raw);
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      contents.push({ role: 'function', parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } }] });
      continue;
    }
    const parts = geminiPartsOfOpenAIMessage(raw);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { model, contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const toolConfig = geminiToolConfig(body.tool_choice);
  if (toolConfig !== null) out.toolConfig = toolConfig;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  out.generationConfig = geminiGenerationConfig(maxTokens, body.temperature, body.top_p, stopList(body.stop), format === 'json_object' || format === 'json_schema');
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the system prompt becomes the
 * systemInstruction, messages become contents (images as inlineData, tool_use as functionCall,
 * tool_result as functionResponse), tools become functionDeclarations, and the output cap, temperature,
 * top_p and stop sequences become the generationConfig. The Anthropic wire carries no JSON mode.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const parts = geminiPartsOfAnthropicMessage(raw);
    if (parts.length === 0) continue;
    contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const out: Json = { model, contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const toolConfig = geminiToolConfig(body.tool_choice);
  if (toolConfig !== null) out.toolConfig = toolConfig;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : undefined;
  out.generationConfig = geminiGenerationConfig(maxTokens, body.temperature, body.top_p, stopList(body.stop_sequences), false);
  return out;
}

/** The text of a Gemini candidate's content parts, joined exactly. */
function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini generateContent answer as a Chat Completions answer naming `model`. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(usage.promptTokenCount);
  const output = numberOrZero(usage.candidatesTokenCount);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(body) }, finish_reason: finishReasonOfGemini(candidate.finishReason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const text = geminiTextOf(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: stopReasonOfGemini(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** One Gemini stream frame's text and, when present, its finish reason and usage. */
function geminiFrameOf(data: unknown): { text: string; finish: unknown; input: number; output: number; hasUsage: boolean } {
  const out = { text: '', finish: undefined as unknown, input: 0, output: 0, hasUsage: false };
  if (!isRecord(data)) return out;
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  out.text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  if (typeof candidate.finishReason === 'string') out.finish = candidate.finishReason;
  if (isRecord(data.usageMetadata)) {
    out.input = numberOrZero(data.usageMetadata.promptTokenCount);
    out.output = numberOrZero(data.usageMetadata.candidatesTokenCount);
    out.hasUsage = true;
  }
  return out;
}

/**
 * A Gemini streamGenerateContent SSE stream as Chat Completions chunks: the role chunk, the text
 * deltas, then one finish chunk, one usage chunk and [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let sentRole = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: unknown = null;
  const emitRole = (): string[] => {
    if (sentRole) return [];
    sentRole = true;
    return [`data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null, null))}\n\n`];
  };
  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    const parsed = geminiFrameOf(data);
    if (parsed.text !== '') {
      yield* emitRole();
      yield `data: ${JSON.stringify(openAIChunk(model, { content: parsed.text }, null, null))}\n\n`;
    }
    if (parsed.finish !== undefined) finish = parsed.finish;
    if (parsed.hasUsage) {
      inputTokens = parsed.input;
      outputTokens = parsed.output;
    }
  }
  yield* emitRole();
  yield `data: ${JSON.stringify(openAIChunk(model, {}, finishReasonOfGemini(finish ?? 'STOP'), null))}\n\n`;
  yield `data: ${JSON.stringify(openAIChunk(model, {}, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }))}\n\n`;
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini streamGenerateContent SSE stream as the Anthropic event sequence: message_start, one text
 * block with the deltas, message_delta with the stop reason and output tokens, then message_stop. The
 * input count is only known at the stream's end, so message_start carries 0; the stream usage check
 * compares the output count alone.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let outputTokens = 0;
  let finish: unknown = null;
  yield sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
  });
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    const parsed = geminiFrameOf(data);
    if (parsed.text !== '') {
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: parsed.text } });
    }
    if (parsed.finish !== undefined) finish = parsed.finish;
    if (parsed.hasUsage) outputTokens = parsed.output;
  }
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReasonOfGemini(finish ?? 'STOP'), stop_sequence: null }, usage: { output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
