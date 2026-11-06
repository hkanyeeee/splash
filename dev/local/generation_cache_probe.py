#!/usr/bin/env python3
"""Check a long answer and its follow-up, recording cache, latency and pressure."""

import argparse
import hashlib
import http.client
import json
import random
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def request(path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=2400)
        try:
            conn.request(
                "GET" if body is None else "POST",
                path,
                None if body is None else json.dumps(body),
                {"Content-Type": "application/json"},
            )
            reply = conn.getresponse()
            data = json.loads(reply.read())
            if reply.status != 200:
                raise RuntimeError(f"{path}: {reply.status}: {data}")
            return data
        finally:
            conn.close()

    model = request("/v1/models")["data"][0]["id"]
    rng = random.Random(20261106)
    words = (
        "river mountain copper garden maple cloud silver meadow bridge lantern".split()
    )
    records = [
        f"Record {n:06d}: {' '.join(rng.choices(words, k=14))}.\n"
        for n in range(args.tokens // 8 + 100)
    ]
    codes = ["CORAL-5821", "MAPLE-9374", "ORBIT-2608"]

    def messages(n):
        return [
            {"role": "system", "content": "Follow the requested JSON format exactly."},
            {
                "role": "user",
                "content": "Keep this archive for a later question. Secret A: "
                + codes[0]
                + "\n"
                + "".join(records[: n // 2])
                + "Secret B: "
                + codes[1]
                + "\n"
                + "".join(records[n // 2 : n])
                + "Secret C: "
                + codes[2]
                + "\nFor this reply, copy the following JSON array exactly. Include every number. "
                "Output the array only, without code fences or explanation:\n"
                + json.dumps(list(range(args.rows))),
            },
        ]

    def count(msgs):
        return request(
            "/v1/chat/completions/count_tokens",
            {"model": model, "messages": msgs, "reasoning_effort": "none"},
        )["input_tokens"]

    low, high = 0, len(records)
    while low < high:
        mid = (low + high + 1) // 2
        if count(messages(mid)) <= args.tokens:
            low = mid
        else:
            high = mid - 1
    msgs = messages(low)
    print(f"prepared {count(msgs)} prompt tokens", flush=True)
    stopped = threading.Event()

    def monitor():
        with (args.output / "memory.jsonl").open("w") as output:
            while not stopped.is_set():
                try:
                    status = request("/status")
                    sample = {
                        "time": time.time(),
                        "memory": status["memory_governor"],
                        "pressure": status["memory_pressure"],
                        "requests": status["active_requests"],
                    }
                    output.write(json.dumps(sample) + "\n")
                    output.flush()
                except (OSError, RuntimeError) as error:
                    output.write(json.dumps({"error": str(error)}) + "\n")
                stopped.wait(5)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    results = []
    try:
        for phase in ("answer", "followup"):
            before = request("/status")
            if before["active_requests"]:
                raise RuntimeError("another request is active")
            body = {
                "model": model,
                "messages": msgs,
                "temperature": 0,
                "seed": 42,
                "reasoning_effort": "none",
                "max_tokens": args.rows * 8 + 128 if phase == "answer" else 128,
                "cache_salt": args.salt,
            }
            start = time.perf_counter()
            response = request("/v1/chat/completions", body)
            elapsed = time.perf_counter() - start
            after = request("/status")
            content = response["choices"][0]["message"]["content"]
            expected = list(range(args.rows)) if phase == "answer" else codes
            try:
                correct = json.loads(content) == expected
            except json.JSONDecodeError:
                correct = False
            result = {
                "phase": phase,
                "seconds": elapsed,
                "correct": correct,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "usage": response["usage"],
                "timings": response.get("timings"),
                "new_checkpoints": after["cache"]["checkpoint_publications"]
                - before["cache"]["checkpoint_publications"],
                "other_submissions": after["requests"]["submitted"]
                - before["requests"]["submitted"]
                - 1,
            }
            for name, data in (
                ("request", body),
                ("response", response),
                ("before", before),
                ("after", after),
            ):
                (args.output / f"{phase}-{name}.json").write_text(
                    json.dumps(data, indent=2)
                )
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
            if not correct or result["other_submissions"]:
                raise RuntimeError("wrong answer or concurrent request")
            if phase == "answer" and response["usage"]["completion_tokens"] < 256:
                raise RuntimeError("answer too short to exercise generated checkpoints")
            msgs += [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "Return only the three secret codes A, B, C as a JSON array, in that order.",
                },
            ]
    finally:
        stopped.set()
        worker.join(timeout=10)


if __name__ == "__main__":
    main()
