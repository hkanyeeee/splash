"""Local Qwen3.8 Q6 benchmark; freeze prompts before comparing builds.

Run with Splash's Python (which supplies tokenizers). Results include complete
responses and status snapshots. This measures latency/acceptance, not task quality.
Example: --prompts build/m4-prompts --output build/m4-results/long-stock
         --tokenizer /path/to/tokenizer.json --lengths 65536,131072,196608
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--lengths", default="65536,131072,196608")
    parser.add_argument("--select", help="run only these comma-separated lengths from the frozen prompts")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--status-port", type=int, default=2235)
    parser.add_argument("--model", default="qwen3.8-27b-mtp")
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()
    args.prompts.mkdir(parents=True, exist_ok=True)
    # Do not overwrite evidence from an earlier run.
    args.output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, data=None):
        port = args.status_port if path == "/status" else args.port
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=3600) as response:
            return json.load(response)

    manifest_path = args.prompts / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["lengths"] != args.lengths:
            raise ValueError("use another prompt directory for different lengths")
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        root = Path(__file__).resolve().parents[2]
        files = sorted((root / "runtime").rglob("*.cpp"))
        corpus = "\n".join(
            f"\nFile: {p.relative_to(root)}\n{p.read_text()}" for p in files
        )
        corpus_ids = tokenizer.encode(corpus, add_special_tokens=False).ids
        manifest = {"lengths": args.lengths, "files": {}}
        for length in map(int, args.lengths.split(",")):
            # Repeat only if the checkout has less source than the requested size.
            ids = (corpus_ids * (length // len(corpus_ids) + 1))[:length]
            content = (
                f"Local performance fixture {length}. Review the following C++ inference runtime.\n"
                + tokenizer.decode(ids)
                + "\n请用中文分析这个推理引擎的内存生命周期、并发调度和取消路径。"
                "指出三个具体风险，引用函数名并给出修复思路与验证方法。"
            )
            name = f"context-{length}.json"
            save(args.prompts / name, [{"role": "user", "content": content}])
            manifest["files"][name] = hashlib.sha256((args.prompts / name).read_bytes()).hexdigest()
        save(manifest_path, manifest)
    save(args.output / "prompts-manifest.json", manifest)
    selected = None if not args.select else {f"context-{int(n)}.json" for n in args.select.split(",")}
    if selected is not None and not selected.issubset(manifest["files"]):
        raise ValueError("selected lengths must exist in the frozen prompt manifest")
    results = []
    for name, digest in manifest["files"].items():
        if selected is not None and name not in selected:
            continue
        raw = (args.prompts / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"prompt changed: {name}")
        messages = json.loads(raw)
        for phase in ("first", "replay", "append"):
            case = f"{Path(name).stem}-{phase}"
            if phase == "append":
                messages = messages + [
                    {"role": "assistant", "content": "我会逐项检查内存所有权与取消后的资源释放。"},
                    {"role": "user", "content": "进一步分析 GPU 异步工作尚未结束时的缓存回收，给出具体的测试程序和预期结果。"},
                ]
            before = request("/status")
            if before["active_requests"]:
                raise RuntimeError("another inference request is active")
            payload = {"model": args.model, "messages": messages,
                       "max_tokens": args.max_tokens, "temperature": 0,
                       "seed": 42, "reasoning_effort": "xhigh"}
            started = time.perf_counter()
            response = request("/v1/chat/completions", payload)
            wall = time.perf_counter() - started
            after = request("/status")
            record = {"case": case, "wall_s": wall, "response": response,
                      "before": before, "after": after,
                      "other_submissions": after["requests"]["submitted"] - before["requests"]["submitted"] - 1}
            save(args.output / f"{case}.json", record)
            if "choices" not in response:
                raise RuntimeError(response)
            summary = {k: record[k] for k in ("case", "wall_s", "other_submissions")}
            summary.update({"timings": response.get("timings"), "usage": response.get("usage"),
                            "speculation": response.get("metrics", {}).get("speculation")})
            results.append(summary)
            save(args.output / "summary.json", results)
            print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
