#!/usr/bin/env python3
"""Measure a cold chat and edits of its last message; keep all HTTP evidence."""

import argparse
import http.client
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def request(path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=1200)
        conn.request(
            "GET" if body is None else "POST",
            path,
            None if body is None else json.dumps(body),
            {"Content-Type": "application/json"},
        )
        reply = conn.getresponse()
        data = json.loads(reply.read())
        assert reply.status == 200, (reply.status, data)
        conn.close()
        return data

    model = request("/v1/models")["data"][0]["id"]
    rows = [
        f"Record {n:05d}: river copper garden cloud window silver meadow bridge.\n"
        for n in range(1000)
    ]

    def messages(n, question):
        return [
            {
                "role": "user",
                "content": "Remember this archive. Secret A: CORAL-5821.\n"
                + "".join(rows[:n])
                + "\nSecret B: ORBIT-2608.",
            },
            {"role": "assistant", "content": "Archive received."},
            {"role": "user", "content": question},
        ]

    questions = [
        "Extract secret A and output only its code. " + "Use the archive above. " * 24,
        "Find secret B and output only its code. "
        + "Read the preceding archive. " * 24,
    ]
    low, high = 0, len(rows)
    while low < high:
        mid = (low + high + 1) // 2
        count = request(
            "/v1/chat/completions/count_tokens",
            {
                "model": model,
                "messages": messages(mid, questions[0]),
                "reasoning_effort": "none",
            },
        )
        if count["input_tokens"] <= 7800:
            low = mid
        else:
            high = mid - 1
    results = []
    for trial in range(2):
        for index, question in enumerate(questions):
            body = {
                "model": model,
                "messages": messages(low, question),
                "temperature": 0,
                "reasoning_effort": "none",
                "max_tokens": 64,
                "cache_salt": f"{args.salt}-{trial}",
            }
            before = request("/status")
            started = time.monotonic()
            reply = request("/v1/chat/completions", body)
            seconds = time.monotonic() - started
            text = reply["choices"][0]["message"].get("content", "")
            expected = ["CORAL-5821", "ORBIT-2608"][index]
            row = {
                "trial": trial,
                "phase": "cold" if index == 0 else "edited_message",
                "seconds": seconds,
                "correct": expected in text,
                "request": body,
                "response": reply,
                "before": before,
                "after": request("/status"),
            }
            results.append(row)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print(
                json.dumps(
                    {k: row[k] for k in ["trial", "phase", "seconds", "correct"]}
                ),
                reply["usage"],
                flush=True,
            )
            assert row["correct"], text


if __name__ == "__main__":
    main()
