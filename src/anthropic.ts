/**
 * ClawRoute Anthropic Adapter
 *
 * Translates between the OpenAI chat-completions format ClawRoute speaks to its
 * clients and Anthropic's native Messages API, so `anthropic/*` models can be
 * routed to directly (no OpenRouter needed).
 */

import { ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ContentPart } from './types.js';
import { safeJsonParse } from './utils.js';

const DEFAULT_MAX_TOKENS = 16000;

const EFFORT_LEVELS = new Set(['low', 'medium', 'high', 'xhigh', 'max']);

/**
 * Models that reject non-default sampling parameters (temperature/top_p/top_k)
 * with a 400: the 5.x generation, Fable/Mythos, and Opus 4.7+.
 */
const NO_SAMPLING_PARAMS = /(-5(-\d+)?$|fable|mythos|opus-4-[78])/;

type AnthropicBlock = Record<string, unknown>;

interface AnthropicMessage {
    role: 'user' | 'assistant';
    content: AnthropicBlock[];
}

/**
 * Build an Anthropic Messages API request body from an OpenAI-format request.
 *
 * @param request - The OpenAI-format request
 * @param modelName - The Anthropic model name (no provider prefix)
 * @returns The Anthropic request body
 */
export function toAnthropicRequest(
    request: ChatCompletionRequest,
    modelName: string
): Record<string, unknown> {
    const systemParts: string[] = [];
    const messages: AnthropicMessage[] = [];

    const push = (role: 'user' | 'assistant', blocks: AnthropicBlock[]): void => {
        if (blocks.length === 0) return;
        const last = messages[messages.length - 1];
        // Anthropic wants tool_results for one assistant turn in a single user message
        if (last && last.role === role) {
            last.content.push(...blocks);
        } else {
            messages.push({ role, content: blocks });
        }
    };

    for (const message of request.messages) {
        if (message.role === 'system') {
            const text = contentToText(message.content);
            if (text) systemParts.push(text);
        } else if (message.role === 'tool') {
            push('user', [
                {
                    type: 'tool_result',
                    tool_use_id: message.tool_call_id ?? '',
                    content: contentToText(message.content),
                },
            ]);
        } else if (message.role === 'assistant') {
            push('assistant', assistantBlocks(message));
        } else {
            push('user', userBlocks(message.content));
        }
    }

    // Assistant prefill is rejected (400) by the current models; drop a trailing text-only turn
    const tail = messages[messages.length - 1];
    if (tail && tail.role === 'assistant' && !tail.content.some((b) => b.type === 'tool_use')) {
        messages.pop();
    }

    const body: Record<string, unknown> = {
        model: modelName,
        max_tokens: positiveInt(request.max_completion_tokens) ?? positiveInt(request.max_tokens) ?? DEFAULT_MAX_TOKENS,
        messages,
    };

    if (systemParts.length > 0) body.system = systemParts.join('\n\n');
    if (request.stream) body.stream = true;

    if (!NO_SAMPLING_PARAMS.test(modelName)) {
        if (typeof request.temperature === 'number') body.temperature = Math.min(request.temperature, 1);
        if (typeof request.top_p === 'number') body.top_p = request.top_p;
    }

    const stop = request.stop;
    if (typeof stop === 'string') body.stop_sequences = [stop];
    else if (Array.isArray(stop) && stop.length > 0) body.stop_sequences = stop;

    if (request.tools && request.tools.length > 0) {
        body.tools = request.tools.map((tool) => ({
            name: tool.function.name,
            description: tool.function.description ?? '',
            input_schema: tool.function.parameters ?? { type: 'object', properties: {} },
        }));
        // Forced tool_choice ("required" / a named function) is a 400 on the current
        // models, so everything except "none" is left to Anthropic's default (auto).
        if (request.tool_choice === 'none') body.tool_choice = { type: 'none' };
    }

    const effort = request.reasoning_effort;
    if (typeof effort === 'string' && EFFORT_LEVELS.has(effort)) {
        body.output_config = { effort };
    }

    return body;
}

/**
 * Convert an Anthropic Messages API response into an OpenAI chat completion.
 *
 * @param data - The parsed Anthropic response
 * @param requestedModel - The model name to report back
 * @returns OpenAI-format chat completion
 */
export function fromAnthropicResponse(
    data: Record<string, unknown>,
    requestedModel: string
): ChatCompletionResponse {
    const blocks = Array.isArray(data.content) ? (data.content as AnthropicBlock[]) : [];

    const text = blocks
        .filter((b) => b.type === 'text')
        .map((b) => String(b.text ?? ''))
        .join('');

    const toolCalls = blocks
        .filter((b) => b.type === 'tool_use')
        .map((b) => ({
            id: String(b.id ?? ''),
            type: 'function' as const,
            function: {
                name: String(b.name ?? ''),
                arguments: JSON.stringify(b.input ?? {}),
            },
        }));

    const message: ChatMessage = { role: 'assistant', content: text || null };
    if (toolCalls.length > 0) message.tool_calls = toolCalls;

    const { promptTokens, completionTokens } = usageOf(data.usage);

    return {
        id: String(data.id ?? `chatcmpl-${Date.now()}`),
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: requestedModel,
        choices: [
            {
                index: 0,
                message,
                finish_reason: mapStopReason(data.stop_reason),
            },
        ],
        usage: {
            prompt_tokens: promptTokens,
            completion_tokens: completionTokens,
            total_tokens: promptTokens + completionTokens,
        },
    };
}

/**
 * Convert an Anthropic error body into an OpenAI-style error body.
 */
export function fromAnthropicError(raw: string, status: number): Record<string, unknown> {
    const parsed = safeJsonParse<{ error?: { type?: string; message?: string } }>(raw);
    return {
        error: {
            message: parsed?.error?.message ?? (raw || `Anthropic request failed (${status})`),
            type: parsed?.error?.type ?? 'api_error',
            code: parsed?.error?.type ?? `http_${status}`,
        },
    };
}

/**
 * Translate an Anthropic SSE stream into OpenAI chat.completion.chunk SSE.
 *
 * @param upstream - The upstream Anthropic streaming body
 * @param requestedModel - The model name to report in chunks
 * @returns A readable stream of OpenAI-format SSE
 */
export function translateAnthropicStream(
    upstream: ReadableStream<Uint8Array>,
    requestedModel: string
): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    let buffer = '';
    let id = `chatcmpl-${Date.now()}`;
    let promptTokens = 0;
    let completionTokens = 0;
    let nextToolIndex = 0;
    let sawToolUse = false;
    let finished = false;
    const toolIndexByBlock = new Map<number, number>();

    const transform = new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, controller) {
            buffer += decoder.decode(chunk, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';
            for (const line of lines) handleLine(line, controller);
        },
        flush(controller) {
            if (buffer.trim()) handleLine(buffer, controller);
            if (!finished) controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        },
    });

    const emit = (
        controller: TransformStreamDefaultController<Uint8Array>,
        delta: Record<string, unknown>,
        finishReason: string | null = null,
        usage?: Record<string, number>
    ): void => {
        const payload: Record<string, unknown> = {
            id,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: requestedModel,
            choices: [{ index: 0, delta, finish_reason: finishReason }],
        };
        if (usage) payload.usage = usage;
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
    };

    function handleLine(line: string, controller: TransformStreamDefaultController<Uint8Array>): void {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) return;

        const event = safeJsonParse<Record<string, any>>(trimmed.slice(5).trim());
        if (!event) return;

        switch (event.type) {
            case 'message_start': {
                id = event.message?.id ?? id;
                promptTokens = usageOf(event.message?.usage).promptTokens;
                emit(controller, { role: 'assistant', content: '' });
                break;
            }
            case 'content_block_start': {
                const block = event.content_block;
                if (block?.type === 'tool_use') {
                    sawToolUse = true;
                    const toolIndex = nextToolIndex++;
                    toolIndexByBlock.set(event.index, toolIndex);
                    emit(controller, {
                        tool_calls: [
                            {
                                index: toolIndex,
                                id: block.id,
                                type: 'function',
                                function: { name: block.name, arguments: '' },
                            },
                        ],
                    });
                }
                break;
            }
            case 'content_block_delta': {
                const delta = event.delta;
                if (delta?.type === 'text_delta') {
                    emit(controller, { content: delta.text });
                } else if (delta?.type === 'input_json_delta') {
                    const toolIndex = toolIndexByBlock.get(event.index);
                    if (toolIndex !== undefined) {
                        emit(controller, {
                            tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }],
                        });
                    }
                }
                break;
            }
            case 'message_delta': {
                completionTokens = event.usage?.output_tokens ?? completionTokens;
                const reason = mapStopReason(event.delta?.stop_reason, sawToolUse);
                emit(controller, {}, reason, {
                    prompt_tokens: promptTokens,
                    completion_tokens: completionTokens,
                    total_tokens: promptTokens + completionTokens,
                });
                break;
            }
            case 'message_stop': {
                finished = true;
                controller.enqueue(encoder.encode('data: [DONE]\n\n'));
                break;
            }
            case 'error': {
                finished = true;
                const error = {
                    error: {
                        message: event.error?.message ?? 'Upstream stream error',
                        type: event.error?.type ?? 'api_error',
                        code: event.error?.type ?? 'stream_error',
                    },
                };
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(error)}\n\n`));
                controller.enqueue(encoder.encode('data: [DONE]\n\n'));
                break;
            }
            default:
                break; // ping, content_block_stop
        }
    }

    return upstream.pipeThrough(transform);
}

// === Helpers ===

function mapStopReason(reason: unknown, sawToolUse = false): string {
    switch (reason) {
        case 'max_tokens':
            return 'length';
        case 'tool_use':
            return 'tool_calls';
        case 'refusal':
            return 'content_filter';
        case 'end_turn':
        case 'stop_sequence':
        case 'pause_turn':
            return 'stop';
        default:
            return sawToolUse ? 'tool_calls' : 'stop';
    }
}

function usageOf(usage: unknown): { promptTokens: number; completionTokens: number } {
    const u = (usage ?? {}) as Record<string, number | undefined>;
    return {
        promptTokens:
            (u.input_tokens ?? 0) + (u.cache_read_input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0),
        completionTokens: u.output_tokens ?? 0,
    };
}

function positiveInt(value: unknown): number | undefined {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

function contentToText(content: string | null | ContentPart[]): string {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    return content
        .filter((p) => p.type === 'text')
        .map((p) => p.text ?? '')
        .join('\n');
}

function userBlocks(content: string | null | ContentPart[]): AnthropicBlock[] {
    if (typeof content === 'string') {
        return content ? [{ type: 'text', text: content }] : [];
    }
    if (!Array.isArray(content)) return [];

    const blocks: AnthropicBlock[] = [];
    for (const part of content) {
        if (part.type === 'text' && part.text) {
            blocks.push({ type: 'text', text: part.text });
        } else if (part.type === 'image_url' && part.image_url?.url) {
            blocks.push({ type: 'image', source: imageSource(part.image_url.url) });
        }
    }
    return blocks;
}

function imageSource(url: string): AnthropicBlock {
    const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (match) {
        return { type: 'base64', media_type: match[1], data: match[2] };
    }
    return { type: 'url', url };
}

function assistantBlocks(message: ChatMessage): AnthropicBlock[] {
    const blocks: AnthropicBlock[] = [];
    const text = contentToText(message.content);
    if (text) blocks.push({ type: 'text', text });

    for (const call of message.tool_calls ?? []) {
        blocks.push({
            type: 'tool_use',
            id: call.id,
            name: call.function.name,
            input: safeJsonParse<Record<string, unknown>>(call.function.arguments || '{}') ?? {},
        });
    }
    return blocks;
}
