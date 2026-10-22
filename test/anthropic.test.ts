import { describe, it, expect } from 'vitest';
import {
    toAnthropicRequest,
    fromAnthropicResponse,
    fromAnthropicError,
    translateAnthropicStream,
} from '../src/anthropic.js';
import { ChatCompletionRequest } from '../src/types.js';

function req(overrides: Partial<ChatCompletionRequest> = {}): ChatCompletionRequest {
    return {
        model: 'anthropic/claude-sonnet-5-5',
        messages: [{ role: 'user', content: 'hi' }],
        ...overrides,
    };
}

describe('toAnthropicRequest', () => {
    it('hoists system messages and defaults max_tokens', () => {
        const body = toAnthropicRequest(
            req({
                messages: [
                    { role: 'system', content: 'Be brief.' },
                    { role: 'user', content: 'hi' },
                ],
            }),
            'claude-sonnet-5-5'
        );
        expect(body.system).toBe('Be brief.');
        expect(body.max_tokens).toBe(16000);
        expect(body.messages).toEqual([{ role: 'user', content: [{ type: 'text', text: 'hi' }] }]);
    });

    it('converts tool calls and merges tool results into one user turn', () => {
        const body = toAnthropicRequest(
            req({
                messages: [
                    { role: 'user', content: 'go' },
                    {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                            { id: 'a', type: 'function', function: { name: 'f', arguments: '{"x":1}' } },
                            { id: 'b', type: 'function', function: { name: 'g', arguments: '{}' } },
                        ],
                    },
                    { role: 'tool', tool_call_id: 'a', content: 'ra' },
                    { role: 'tool', tool_call_id: 'b', content: 'rb' },
                ],
                tools: [{ type: 'function', function: { name: 'f', parameters: { type: 'object' } } }],
            }),
            'claude-sonnet-5-5'
        );
        const msgs = body.messages as Array<{ role: string; content: Array<Record<string, unknown>> }>;
        expect(msgs).toHaveLength(3);
        expect(msgs[1]?.content[0]).toMatchObject({ type: 'tool_use', id: 'a', input: { x: 1 } });
        expect(msgs[2]?.content).toHaveLength(2);
        expect(msgs[2]?.content[0]).toMatchObject({ type: 'tool_result', tool_use_id: 'a' });
    });

    it('drops forced tool_choice, sampling params and trailing prefill on new models', () => {
        const body = toAnthropicRequest(
            req({
                temperature: 0.2,
                tool_choice: 'required',
                tools: [{ type: 'function', function: { name: 'f' } }],
                messages: [
                    { role: 'user', content: 'hi' },
                    { role: 'assistant', content: '{' },
                ],
            }),
            'claude-opus-5-5'
        );
        expect(body.tool_choice).toBeUndefined();
        expect(body.temperature).toBeUndefined();
        expect((body.messages as unknown[]).length).toBe(1);
    });

    it('keeps temperature on older models and maps images', () => {
        const body = toAnthropicRequest(
            req({
                temperature: 0.2,
                messages: [
                    {
                        role: 'user',
                        content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AAA' } }],
                    },
                ],
            }),
            'claude-sonnet-4-6'
        );
        expect(body.temperature).toBe(0.2);
        const msgs = body.messages as Array<{ content: Array<Record<string, unknown>> }>;
        expect(msgs[0]?.content[0]).toEqual({
            type: 'image',
            source: { type: 'base64', media_type: 'image/png', data: 'AAA' },
        });
    });
});

describe('fromAnthropicResponse', () => {
    it('maps text, tool_use, stop reason and usage', () => {
        const out = fromAnthropicResponse(
            {
                id: 'msg_1',
                stop_reason: 'tool_use',
                content: [
                    { type: 'text', text: 'ok' },
                    { type: 'tool_use', id: 't1', name: 'f', input: { a: 1 } },
                ],
                usage: { input_tokens: 10, cache_read_input_tokens: 5, output_tokens: 7 },
            },
            'claude-sonnet-5-5'
        );
        const choice = out.choices[0]!;
        expect(choice.finish_reason).toBe('tool_calls');
        expect(choice.message.content).toBe('ok');
        expect(choice.message.tool_calls?.[0]?.function).toEqual({ name: 'f', arguments: '{"a":1}' });
        expect(out.usage).toEqual({ prompt_tokens: 15, completion_tokens: 7, total_tokens: 22 });
    });

    it('converts error bodies', () => {
        const err = fromAnthropicError(
            '{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}',
            400
        ) as { error: { message: string; type: string } };
        expect(err.error.message).toBe('bad');
        expect(err.error.type).toBe('invalid_request_error');
    });
});

describe('translateAnthropicStream', () => {
    it('emits OpenAI chunks, tool deltas, usage and [DONE]', async () => {
        const events = [
            { type: 'message_start', message: { id: 'msg_1', usage: { input_tokens: 4 } } },
            { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hel' } },
            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'lo' } },
            { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 't1', name: 'f' } },
            { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"a":1}' } },
            { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 9 } },
            { type: 'message_stop' },
        ];
        const sse = events.map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join('');
        const upstream = new Response(sse).body!;

        const text = await new Response(translateAnthropicStream(upstream, 'claude-sonnet-5-5')).text();
        const chunks = text
            .split('\n\n')
            .filter((l) => l.startsWith('data: ') && !l.includes('[DONE]'))
            .map((l) => JSON.parse(l.slice(6)));

        const content = chunks.map((c) => c.choices[0].delta.content ?? '').join('');
        expect(content).toBe('Hello');
        expect(chunks.some((c) => c.choices[0].delta.tool_calls?.[0]?.function?.name === 'f')).toBe(true);
        const last = chunks[chunks.length - 1];
        expect(last.choices[0].finish_reason).toBe('tool_calls');
        expect(last.usage).toEqual({ prompt_tokens: 4, completion_tokens: 9, total_tokens: 13 });
        expect(text.trimEnd().endsWith('data: [DONE]')).toBe(true);
    });
});
