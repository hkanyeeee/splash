r"""Prepare fixed 100K, 150K and 208K chat inputs, then replay them against Splash.

    python dev/benchmarks/long_context.py prepare INPUTS --tokenizer tokenizer.json \
        --model qwen3.6-35b-a3b --revision f61c31db
    python dev/benchmarks/long_context.py run INPUTS RESULTS --url http://127.0.0.1:8080

Use the same prepared directory for both builds and a fresh server cache per run.
Pass --size to run a single case on a freshly started server.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import tarfile
import urllib.request


SIZES = (100_000, 150_000, 208_000)
SUFFIXES = {".md", ".py", ".cpp", ".hpp", ".mm", ".metal", ".h"}


def prepare(args):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", args.revision + "^{commit}"], text=True
    ).strip()
    archive = subprocess.check_output(["git", "archive", revision, "--", "runtime", "server", "docs"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        sources = {
            file.name: tree.extractfile(file).read().decode()
            for file in tree if file.isfile() and Path(file.name).suffix in SUFFIXES
        }
    args.inputs.mkdir(parents=True, exist_ok=False)
    for size in SIZES:
        names = sorted(sources)
        random.Random(size).shuffle(names)
        text = "".join("\nFILE " + name + "\n" + sources[name] for name in names)
        tokens = tokenizer.encode(text[:size * 6], add_special_tokens=False).ids
        if len(tokens) < size:
            raise ValueError(f"source revision {revision} provides too few tokens for {size}")
        boundaries = (0, round(size * .06), size // 2, round(size * .94), size - 145)
        parts = [tokenizer.decode(tokens[a:b], skip_special_tokens=False) for a, b in zip(boundaries, boundaries[1:])]
        keys = [hashlib.sha256(f"{size}-{i}".encode()).hexdigest()[:10] for i in range(3)]
        prompt = "Inspect these implementation excerpts and their three review records.\n" + parts[0]
        for i, (key, part) in enumerate(zip(keys, parts[1:]), 1):
            prompt += f"\n[REVIEW RECORD {i} reference {key}]\n" + part
        prompt += ("\nList the three review record references in order, then describe the inference engine "
                   "responsibilities illustrated in the excerpts. Use complete sentences and identify any "
                   "two implementation details.")
        request = dict(model=args.model, messages=[dict(role="user", content=prompt)], temperature=0,
                       seed=42, reasoning_effort="none", max_tokens=128)
        (args.inputs / f"{size}.json").write_text(json.dumps(request, ensure_ascii=False))
        (args.inputs / f"{size}-key.json").write_text(json.dumps(keys))


def run(args):
    args.results.mkdir(parents=True, exist_ok=True)
    for size in ([args.size] if args.size else SIZES):
        payload = (args.inputs / f"{size}.json").read_bytes()
        keys = json.loads((args.inputs / f"{size}-key.json").read_text())
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", data=payload,
                                         headers={"Content-Type": "application/json"})
        path = args.results / f"{size}-response.json"
        if path.exists():
            raise FileExistsError(path)
        with urllib.request.urlopen(request, timeout=7200) as response:
            raw = response.read()
        with path.open("xb") as f:
            f.write(raw)
        result = json.loads(raw)
        t = result["timings"]
        text = result["choices"][0]["message"]["content"]
        positions = [text.find(key) for key in keys]
        in_order = all(pos >= 0 for pos in positions) and positions == sorted(positions)
        print(json.dumps(dict(input_tokens=t["prompt_n"], cached_tokens=t["cache_n"], output_tokens=t["predicted_n"],
                              prefill_tps=t["prompt_per_second"], decode_tps=t["predicted_per_second"],
                              ttft_ms=result["metrics"]["request_latency"]["ttft_ms"], keys_in_order=in_order,
                              request_sha256=hashlib.sha256(payload).hexdigest())), flush=True)
        if t["cache_n"] or abs(t["prompt_n"] - size) >= 200 or t["predicted_n"] != 128 or not in_order:
            raise ValueError(f"{size} failed the cold input, output length or retrieval check; see {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("inputs", type=Path)
    prep.add_argument("--tokenizer", type=Path, required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--revision", default="HEAD")
    prep.set_defaults(action=prepare)
    replay = commands.add_parser("run")
    replay.add_argument("inputs", type=Path)
    replay.add_argument("results", type=Path)
    replay.add_argument("--url", required=True)
    replay.add_argument("--size", type=int, choices=SIZES)
    replay.set_defaults(action=run)
    options = parser.parse_args()
    options.action(options)
