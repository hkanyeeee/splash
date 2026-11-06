# Local M2 Max server

This checkout's `local/m2max-bf16-24g` branch starts from upstream main
`9124ff2c` (Splash 1.6). Its Apple7/8 kernels come from
[paperniuk/splash, `apple7-m1-1.5`](https://github.com/paperniuk/splash/tree/5e0098eda1f7334bc99a5ab5196fcdcaa65f809f),
ported to main's kernel calibration, paged attention and chunked expert interfaces.
The older `origin/local/m4max-q6k` branch is not the base.

The machine is an Apple M2 Max with 30 GPU cores, 32 GiB unified memory and
macOS 27.2. The target remains the existing file:

```
/Users/hewenxin/gguf_models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
```

The DFlash2 draft is `incoai/Qwen3.8-27B-DFlash2`, revision
`015e795645c74b1a0eeef3b570031fb62e769bc5`, in the Hugging Face cache. Startup
uses the prepared local files offline. No copy of the target weights is needed.

## Runtime changes

- Apple7/8 get simdgroup matrix kernels and FP32 activation tables; Apple9/10
  retain main's calibration and selection.
- Dense and sparse attention use register kernels on Apple7/8.
- Prefill ends command buffers between layers on Apple7/8 and splits long
  attention dispatches into bounded pieces.
- The Apple7/8 kernels use main's current projection and expert buffer layouts.
- Host steps use the plain path on Apple7/8, where the GPU spin can race ahead
  of earlier work.

`dev/local/run-server.sh` holds the local arguments. Its default is port 1234,
bound to all IPv4 interfaces as the previous llama-server was. It accepts the
previous client alias `qwen3.8-27b-mtp-98`, and `qwen3.8-27b`; Splash uses DFlash2
speculation even when a client uses the old alias.

The configuration uses a **24 GiB Metal budget, BF16 KV, BF16 recurrent state,
text only, and one admitted request**. KV is not INT8-quantized. ANE prefill
calibration is enabled; the disk prefix cache is disabled. The RAM prefix cache remains active.
Idle weights are released after ten minutes. Additional simultaneous generation
requests receive 503 instead of running another model lane.

`--max-memory` limits Splash's governed allocations, not the sum of process RSS.
Validation records Metal allocations, the native physical footprint and the HTTP
process separately. KV uses BF16 with zero quantization scales. The arenas use
199.6 MB of prefill scratch and 84.1 MB of decode scratch.
A near-limit test exposed missing continuation-state headroom: the old plan
advertised 91,001 tokens but stalled at 90,112 of a 90,778-token prompt while
trying to reserve its next-turn state. Memory planning now leaves one cached
state beside the active lane before sizing the KV pool. This reserve stays
elastic, so the allocator can actually use it; it is also reflected in streamed
model feasibility and status as `continuation_state_bytes`. Regression tests
cover a full KV pool plus that state, the minimum budget and overflow.
At 24 GiB startup calibration chose ANE share 15/34 (44.1%), taking 24.3 ms
per 512-row FFN layer against the GPU alone's 36.0 ms. Its local FFN check
reported 1.5% relative RMS difference on the ANE's part. The resulting context
plan is 102,265 tokens, versus 106,073 with the ANE off. The same cold 3,989-token
prompt with ANE took 22.9 seconds overall (178.8 prefill tokens/s), with correct
retrieval and a 1.36-second cached followup.

The near-limit test passed on 2026-11-06:

| Request | Prompt tokens | Cached tokens | Total time | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Cold archive retrieval | 101,987 | 0 | 904.30 s | 35.90 |
| Cached followup | 102,053 | 101,952 | 2.73 s | 41.99 |

Both replies recovered the codes at the beginning, middle and end of the archive.
Cold prefill took 903.15 seconds (112.92 tokens/s). The limit includes prompt and
generated tokens: a prompt above 102,265 was rejected with HTTP 400 and
`context_length_exceeded`. The test sampled 23.999 GiB of governed allocations,
23.534 GiB of native physical footprint, and 458 MiB of frontend RSS at their
respective peaks. System pressure reached `warning`, with available host memory
as low as 1.86 GiB; thermal state stayed `nominal`. Swap did not grow (388.38 MiB
initially, 380.38 MiB at the end). ANE remained active with zero reruns. This is
a tested near-capacity configuration; memory pressure depends on other running
applications.

A cold 3,989-token BF16 prompt retrieved all three codes in 28.4 seconds:
142.7 prefill tokens/s and 63.0 generated tokens/s. Its followup reused 3,968
tokens and finished in 1.36 seconds. An overlapping generation request was
rejected with HTTP 503 (`frontend_overloaded`), as intended.

The local wrapper exports `SPLASH_RUNTIME_LANES=1` and
`SPLASH_PREFILL_ROWS=512`. These build settings reduce the native execution
limit and prefill scratch; ordinary `make` retains main's four-lane, 2048-row
defaults. Metal's four-slot buffer ABI is preserved, with unused slots bound
to lane zero. Decode arenas still reserve any padding required by a selected
projection tile. Cached draft windows rebuild in pieces without shortening
the draft model's trained 2048-token window. A 512-row ANE profile calibrates
against two different GPU row counts even though it has only one ANE function.
Startup also warms Apple7/8's 16-row and 32-row short-prompt kernels. Their first
Metal compilation otherwise blocks the serving loop long enough to trigger its
30-second watchdog; the full-size and one-row warmups miss these variants.

For comparison runs, `SPLASH_LOCAL_MEMORY=23G` lowers the budget and
`SPLASH_LOCAL_ANE=off` keeps the GPU alone (`auto` enables startup calibration).
ANE computes its share of the FFN with W8A8 weights/activations; KV storage
continues to use BF16 in either case.

## Build and validation

```
make -j4 SPLASH_RUNTIME_LANES=1 SPLASH_PREFILL_ROWS=512
# General CPU suite uses the default execution profile:
make -j2 BUILD=build/default-check test-engine-cpu
make architecture-check
```

The focused Metal checks include `gguf-projection`, `norm-table`, `gdn-decode`,
`q8-flash-attention`, `q8-chunked-prefill` and the backend command-buffer tests.
Run them with `MTL_SHADER_VALIDATION=1`. Local results are retained in
`build/local-validation/`; these artifacts are not committed.
The full CPU suite and architecture check passed. Metal validation passed all
44 projection formats, dense/sparse attention including 131,072-token histories,
chunked prefill and 47 backend checks. Modern-GPU selection remained unchanged
across 2,873 test groups. Memory planning and both the default and local execution
profiles have focused regression coverage.

`dev/local/context_probe.py` uses the HTTP token-count endpoint to construct a
prompt near a requested token count, checks secret-code retrieval at its beginning,
middle and end, and repeats the question in a cached followup. It writes the
requests, responses, timings and status/physical-memory samples. For example:

```
.venv/bin/python dev/local/context_probe.py --port 8000 --tokens 32000 \
  --output build/local-validation/context-32k
```

## Service management

The installed LaunchAgent is `~/Library/LaunchAgents/com.user.splashserver.plist`,
copied from `dev/local/com.user.splashserver.plist`. It starts at login and is
kept alive by launchd. The API base is `http://127.0.0.1:1234/v1`. Logs are
`~/Library/Logs/Splash/server.log` and `server.err`.
Deployment checks covered Chinese SSE output, default medium reasoning, and
explicit memory release followed by a 4K prompt and cached continuation. Weight
restore took 10.92 seconds, ANE program reload 0.03 seconds; resumed prefill was
178.6 tokens/s, with no restore failures or ANE reruns. The first startup after
the short-prompt fix took 175 seconds including Metal compilation. A subsequent
LaunchAgent restart became ready in 17 seconds; the old native process exited,
and the replacement answered a short prompt successfully. The installed agent
is enabled and the original llama-server agent is disabled.

```
launchctl print gui/$(id -u)/com.user.splashserver
launchctl kickstart -k gui/$(id -u)/com.user.splashserver
curl http://127.0.0.1:1234/health
```

The old llama-server agent is stopped and disabled. Its plist and wrapper are
preserved at `~/Library/LaunchAgents/com.user.llamaserver.plist` and
`~/.llama_server.sh`. To switch back:

```
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.user.splashserver.plist
launchctl disable gui/$(id -u)/com.user.splashserver
launchctl enable gui/$(id -u)/com.user.llamaserver
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.llamaserver.plist
```
