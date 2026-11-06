#!/usr/bin/env python3
"""Exercise a local server's advertised context with retrieval and a cached turn.

Writes complete HTTP responses, periodic status/RSS samples and a result JSON.
Run with the checkout's Python; this uses only the standard library.
"""
import argparse
import http.client
import json
from pathlib import Path
import random
import subprocess
import threading
import time


def request(port, path, body=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=7200)
    try:
        connection.request('GET' if body is None else 'POST', path,
                           None if body is None else json.dumps(body),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        data = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f'{path}: HTTP {response.status}: {data}')
        return data
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pids', type=int, nargs='*', default=[])
    parser.add_argument('--message-boundary', action='store_true',
                        help='Put the final question in a separate chat message to exercise its checkpoint')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = request(args.port, '/v1/models')['data'][0]['id']
    rng = random.Random(20261106)
    words = 'river mountain copper garden maple cloud window silver meadow bridge lantern harbor'.split()
    rows = ['Record %06d: %s.\n' % (n, ' '.join(rng.choices(words, k=14)))
            for n in range(args.tokens // 8 + 100)]
    expected = ['CORAL-5821', 'MAPLE-9374', 'ORBIT-2608']

    def messages(n):
        mid = n // 2
        text = ('Read this archive and return the three secret codes only as a JSON array in order A, B, C.\n'
                'Secret A: ' + expected[0] + '\n' + ''.join(rows[:mid]) +
                'Secret B: ' + expected[1] + '\n' + ''.join(rows[mid:n]) +
                'Secret C: ' + expected[2] + '\nWhat are secret codes A, B and C?')
        result = [{'role': 'user', 'content': text}]
        if args.message_boundary:
            result += [{'role': 'assistant', 'content': 'Archive received.'},
                       {'role': 'user', 'content': 'Return the three secret codes A, B and C as a JSON array.'}]
        return result

    def count(msgs):
        return request(args.port, '/v1/chat/completions/count_tokens',
                       {'model': model, 'messages': msgs, 'reasoning_effort': 'none'})['input_tokens']

    low, high = 0, len(rows)
    while low < high:
        mid = (low + high + 1) // 2
        if count(messages(mid)) <= args.tokens:
            low = mid
        else:
            high = mid - 1
    msgs = messages(low)
    actual = count(msgs)
    (args.output / 'request.json').write_text(json.dumps({'model': model, 'messages': msgs}))
    print(f'Prompt prepared: {actual} tokens (requested {args.tokens})', flush=True)
    stopped = threading.Event()
    started = time.monotonic()

    def monitor():
        with (args.output / 'samples.jsonl').open('w') as output:
            while not stopped.is_set():
                sample = {'seconds': time.monotonic() - started}
                try:
                    sample['status'] = request(args.port, '/status')
                    if args.pids:
                        sample['rss_kib'] = subprocess.check_output(
                            ['ps', '-o', 'pid=,rss=', '-p', ','.join(map(str, args.pids))], text=True).strip()
                except Exception as error:
                    sample['error'] = str(error)
                output.write(json.dumps(sample) + '\n')
                output.flush()
                stopped.wait(5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    results = []
    try:
        for label in ['cold', 'cached_followup']:
            t0 = time.monotonic()
            result = request(args.port, '/v1/chat/completions', {
                'model': model, 'messages': msgs, 'temperature': 0,
                'reasoning_effort': 'none', 'max_tokens': 96,
            })
            (args.output / f'{label}.json').write_text(json.dumps(result, indent=2))
            content = result['choices'][0]['message'].get('content', '')
            correct = all(code in content for code in expected)
            row = {'label': label, 'seconds': time.monotonic() - t0,
                   'correct': correct, 'content': content, 'usage': result.get('usage'),
                   'timings': result.get('timings')}
            results.append(row)
            print(json.dumps(row), flush=True)
            if not correct:
                raise RuntimeError(f'{label}: retrieval failed: {content!r}')
            msgs += [{'role': 'assistant', 'content': content},
                     {'role': 'user', 'content': 'Repeat the same three secret codes in the same order.'}]
        cached = results[-1]['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if cached < actual * 0.9:
            raise RuntimeError(f'Expected cached followup to reuse the prompt, got {cached}/{actual}')
    finally:
        stopped.set()
        thread.join(timeout=10)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    print('PASS: retrieval at the beginning, middle and end; cached followup reused the prompt.', flush=True)


if __name__ == '__main__':
    main()
