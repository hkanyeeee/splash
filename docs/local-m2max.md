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
- The local launcher enables generated-prefix checkpoints at natural decode
  boundaries, with optional state allocation admitted by the memory governor.

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
Run them with `MTL_SHADER_VALIDATION=1`. Local checks write temporary artifacts
under `build/local-validation/`; these artifacts are not committed. The recorded
results below survive the 2026-11-06 cleanup of those generated files.
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

## M4 Max branch review

The follow-up review compares `origin/local/m4max-q6k` (`79a42fdd`) against
its base `031a25a6`. It is a selective port onto main. Device guards in that
branch describe where an experiment was enabled; they do not establish that
an earlier GPU cannot execute it. The standalone probes below bypass those
policy guards and execute the kernels on this M2 Max.

| M4 change | Outcome on this M2 deployment |
| --- | --- |
| Final-message checkpoint | Adapted to main's rolling checkpoint and memory governor; template-derived metadata replaces hardcoded vocabulary/token IDs. |
| Keep a message checkpoint on disk under pressure | Not enabled: this deployment has zero disk-cache quota. Main's normal disposable-state reclamation remains in use. |
| Generated-prefix checkpoint and GDN prefix kernel | Not ported. The within-cycle GDN prefix kernel requires FP32 recurrent state and fixed geometry. The separate natural decode-boundary checkpoint has no FP32 guard and is a candidate for BF16 validation; the original review conflated these two paths. See the updated-branch review below. |
| Parallel gate/up kernels | Run correctly on M2 in the tested formats, but the M4 BF16-table implementation loses to the current FP32-table path. Retain the current path. |
| Page32 attention rescale flags | All four M4 variants run correctly on M2; the current register-attention path is faster in every measured case. Retain it. |
| Short-prefill staged64 tuning | Runs on M2, but at 33/48/64 rows the measured Q4_K gate/up takes about 4.77 ms at its best split, versus 1.71 ms on the current prefill tile. Keep the 32-row crossover. |
| 44 GiB override | Not ported: it was for a 48 GiB M4 Max. Current main already honors this machine's explicit 24 GiB budget. |

For text chat prompts of at least 4096 tokens, the frontend renders the history
before the final message without a generation prompt. It sends a hint only
after verifying that both the rendered text and token IDs match the complete
prompt's prefix. Unsupported partial templates and image requests use the
existing policy. The internal protocol is version 13; frontend and native
binary must be upgraded or rolled back together.

The engine rounds the hint down to a KV page. When it is ahead of the last
planned progress checkpoint and at least a draft window beyond the restored
state, it moves that last point to the message boundary. It keeps the existing
rolling state lifetime, pin handling and pressure reclamation. It does not
retain an additional checkpoint beside the usual last checkpoint and replay
state. If a long final message already has later progress checkpoints, the
hint does not replace them. This preserves recovery granularity and the
24 GiB allocation policy.

`dev/local/message_cache_probe.py` records paired cold chats and edits of their
final message. Each pair uses a fresh cache salt, identical prompt content and
generation settings across builds. The unchanged target and draft weights,
BF16 formats, 512-row prefill, single lane and ANE calibration are held fixed.
Raw responses and status snapshots were written under
`build/local-validation/m4-port/` and removed during the cleanup below.

The two paired 7,785-token trials produced identical answers across builds:

| Request | Baseline mean | Message-checkpoint mean | Cached tokens |
| --- | ---: | ---: | ---: |
| Cold conversation | 43.664 s | 43.541 s | 0 in both |
| Replace the final message | 21.462 s | 1.623 s | 4,096 → 7,616 |

The edit reused 3,520 additional tokens. Its latency fell 92.4% in this fixture;
the complete cold-plus-edit pair fell 30.65%. This is avoided prefill work on a
specific edit pattern, not a 13× increase in general model inference speed.
The initial control with only a single differing token at the very end already
hit the baseline replay state; its roughly 0.47-second result was recorded
separately as `baseline-probe`, and is not credited as a benefit of the new feature.

The updated service also passed a 101,993-token cold request and its
102,045-token follow-up, retrieving all three codes from the archive's start,
middle and end. Cold prefill was 112.88 tokens/s; the follow-up reused 101,984
tokens and completed in 1.73 seconds. Five-second samples peaked at 23.940 GiB
charged and 23.512 GiB native footprint, with no capacity failure, Metal
failure, native restart or ANE rerun. Host pressure reached warning and minimum
available memory was 1.71 GiB; swap fell from 380.38 to 372.38 MiB. These results
support the tested single-lane 24 GiB setup, including its existing reclamation
policy, rather than a claim of unlimited host-memory headroom.

### M2 execution of the M4 kernels

Temporary probe sources and full logs were generated under
`build/local-validation/m4-port/kernel-probe/` and removed during cleanup.
The M4 kernels remain available in commit `79a42fdd`. The old M4 sources compile
against the current ABI without algorithm changes, with Metal 4 and the same
macOS 26.4 deployment target as production. An initial attention build that
omitted that target hit a pipeline stack-space error; rebuilding with the
production target made all four variants executable. This was a build-setting
issue, not evidence of a missing M2 hardware feature.

Gate/up probes use eight rows, four K splits, 512×2048 for an FP64 reference
check and 17408×5120 for a model-sized check and timing. Original Q5_K/Q5_K,
Q6_K/Q6_K and Q8_0/Q8_0 M4 kernels produce bitwise-identical output to the BF16-table
two-pass reference at both sizes. Probe-only Q4_K/Q4_K and IQ4_XS/IQ4_XS
instantiations of the same template do too. The latter is this model's most
common gate/up format pair (20 of 64 layers). Timings include table preparation and use six distinct
weight pairs, 20 alternating-order rounds, and GPU command timestamps. These
are operator measurements, not whole-model generation speeds.

| Format | M4 parallel, BF16 table | Current M2, FP32 table | Existing M2 one-pass candidate |
| --- | ---: | ---: | ---: |
| IQ4_XS | 0.550 ms | 0.481 ms | 0.457 ms |
| Q4_K | 0.609 ms | 0.523 ms | 0.504 ms |
| Q5_K | 0.758 ms | 0.625 ms | 0.593 ms |
| Q6_K | 0.764 ms | 0.642 ms | 0.622 ms |
| Q8_0 | 0.717 ms | 0.611 ms | 0.580 ms |

The existing FP32 one-pass implementation also matches the M2 two-pass output
bitwise. Its small operator-level gains do not yet establish an end-to-end
benefit across this model's mixed gate/up formats; this review does not change
its calibration policy.

Attention probes exercise BF16 KV, 24 query heads, four KV heads and dimension
256, with eight verify rows or 33 prefill rows, at histories of 511, 4096 and
32768 tokens. All four M4 variants and both baselines pass the FP64 error bounds
(36 cases). At 32768 tokens, the fastest M4 variant takes 4.099 ms for verify
versus 1.357 ms on the current M2 path, and 16.469 ms for prefill versus 5.264 ms.
At shorter histories the current M2 path also wins. The result distinguishes
hardware capability, numerical correctness and performance: the tested M4
kernels are supported, but are not the best kernels for this M2 deployment.

Validation of the message-checkpoint change includes all 1,064 server tests,
the native CPU suites (including protocol interoperability and checkpoint
resume tests), the paired edit benchmark, and the near-limit retrieval test
above. The kernel probes do not modify the deployed metallib or its selection
policy.

## Disk cleanup (2026-11-06)

After deployment, removed approximately 661 MiB of generated files:

- Duplicate default-profile builds, test executables, benchmark binaries,
  baseline copies, raw prompts and responses, and status samples (463 MiB).
- This session's pip download cache (67 MiB), Metal shader-validation cache
  (105 MiB), and unused ANE calibration programs and old calibration records
  (21 MiB).
- Temporary scripts, patches, stack samples and logs, Python/linter caches,
  and the stopped port-8000 test server's records (5 MiB).

Kept the deployed executable and metallibs with their incremental build inputs,
the Python environment, prepared model/tokenizer metadata, target and DFlash2
weights, the active ANE program and calibration, and the current service records
and logs. Shared system Metal caches that predate this work were left intact.
The figures above are rounded allocated file sizes, rather than a measurement
of APFS physical space reclaimed. Numerical results and implementation tests
remain in Git; the uncommitted raw experiment artifacts have been deleted.

## Updated M4 branch review (2026-11-06)

Fetched `origin/local/m4max-q6k` from `79a42fdd` to `7b4f28f5`. Its common
main ancestor with this checkout is `9124ff2c`; the main code introduced by
the M4 merge is already present here. The three new branch-specific commits
are `6654c3c2` (main adaptation), `2bbbdcf6` (generated-state admission) and
`7b4f28f5` (long-prefix attention reduction). This review reads the implementation
and regression tests; it does not establish new M2 runtime measurements.

The most useful unported feature is **natural-boundary generation caching**.
In `runtime/model/Runtime.cpp`, `keepsGenerationWindow()` checks the dense
model geometry and absence of auxiliary state, without requiring FP32. It
maintains the draft window using only accepted rows, then captures state when
a decode cycle starts on a 32-token boundary, after at least 64 generated
tokens and with at least 128 tokens since its previous generated checkpoint.
This can let a follow-up reuse part of a long assistant answer. Our existing
message-prefix checkpoint instead helps edits to the final input message;
it does not implement generated-prefix publication.

Only `usesAlignedGeneration()` requires FP32: it reconstructs a page boundary
inside a decode cycle with `verify_gdn_prefix_f32`. The natural-boundary path
already existed in `79a42fdd`; overlooking its distinction from the aligned
path was an error in the earlier review, not a newly removed precision guard.
Lack of a guard is not proof of BF16 numerical correctness. A port should first
validate accepted/rejected rows and window wraparound, cancellation, duplicate
publication, restoration, long answers and follow-ups on this M2.

The generated-cache port must include **pre-submission state admission** from
`2bbbdcf6`: `Engine::prepare()` reserves optional checkpoint buffers under the
request's memory claim and passes them to the model. If memory refuses, decoding
continues without the checkpoint. Its regression test uses the real governor
through the fake allocator and checks both acceptance and refusal. This matters
at our 24 GiB limit; copying the earlier allocation inside `decodeAsync()` would
miss main's new allocation contract. Our current prefill capture already uses
engine admission, and we have no generated-cache path exposing that earlier bug.

The attention fix records a **full-model long-prefix regression** when a
288-row tail after 65,536 tokens changed from one to seven reduction splits.
It restores the accepted policy for 40-core Apple9. Our Apple8 register kernels
also have a separate occupancy rule, so copying that device's split constant
would not reproduce its execution. The useful lesson is to require fixed-prompt
full-model output/logit or task-quality comparisons, including these tail shapes,
alongside kernel speed and numerical tests. Our successful 102K retrieval test
is useful coverage but does not establish full-model parity for every split rule.

Other reviewed differences:

- The M4 gate/up and attention kernel files, GDN prefix kernel and
  `LOCAL_OPTIMIZATION.txt` are byte-identical to `79a42fdd`. The old measurements
  do not constitute new-main measurements; the earlier M2 operator comparison
  still applies to these unchanged kernels.
- M4 keeps both an earlier recovery checkpoint and a message checkpoint. Our
  version moves the last progress checkpoint to bound memory, derives its hint
  from the actual template and token prefix, and avoids hardcoded token IDs.
  Retaining both is a memory/coverage tradeoff, not an automatic upgrade at 24 GiB.
- Disk rescue requires a disk tier; this deployment has a zero disk-cache quota.
  The 44 GiB override and staged64 crossover are specific measured policies for
  that deployment. Neither supplies a missing requirement for our current setup.
- The Swift test changes isolate locale, asynchronous page loading and reduced
  motion preferences. They improve test stability, without changing inference.

Recommended next implementation: natural-boundary generated-prefix caching,
with pre-submission admission and graceful refusal, retaining BF16 KV/state and
the 24 GiB limit. Benchmark complete long-answer/follow-up pairs against the
current build, including capture overhead and memory pressure, before deployment.
This audit changes documentation only; the serving implementation stays at
`36edabdc` and its existing measured kernel policy.

## Natural-boundary generation cache (2026-11-06)

Ported the natural decode-boundary cache and pre-submission state admission
from `origin/local/m4max-q6k` at `7b4f28f5`. The local launcher enables it with
`SPLASH_LOCAL_GENERATION_CACHE=1`; set it to `0` for isolated comparisons.
Other launchers retain the disabled default. This path keeps BF16 KV and
recurrent state, the 24 GiB memory limit, one runtime lane, 512-row prefill and
ANE offload. It retains this checkout's M2 kernel choices and template-derived
message checkpoint.

The opt-in is scoped to the validated dense model geometry (248,320 vocabulary,
5,120 hidden width, 48 recurrent layers, no auxiliary state). Accepted draft
rows update the existing compressed context window; rejected proposals do not
overwrite older slots. This uses main's existing draft-window representation;
it does not introduce KV or recurrent-state quantization. At a natural 32-token
page boundary, after at least 64 generated tokens and 128 tokens since the
previous capture, the engine may reserve a new snapshot. The model copies the
old window before submission and seals the corresponding recurrent state after
the GPU completes. Rolling publication replaces an unpinned previous generated
checkpoint. Admission refusal skips the optional capture and allows decoding
to continue. No aligned within-cycle reconstruction or FP32 conversion is used.

Validation before deployment:

- Full default-profile CPU suite: engine 397 passed, zero failed; engine
  simulation and the other CPU suites also passed. Added lifecycle coverage
  checks bounded rolling publication, deduplication, follow-up restoration and
  cancellation. The real governor with a fake allocator checks admitted and
  refused captures produce the same complete output.
- Metal draft-attention suite passed with `MTL_SHADER_VALIDATION=1`. The added
  oracle compares the entire context window against prefill storage for every
  accepted count from 0 through 8, including rejected slots and wraparound at
  positions 0, 2,043, 2,047 and 65,535, across the supported test shapes.
- Architecture check, whitespace check and lint for the reproducible
  `dev/local/generation_cache_probe.py` passed.
- Real API checks also passed for a 128-integer JSON-schema-constrained answer
  (four generated checkpoints; follow-up restored 1,024 tokens), and a streamed
  answer disconnected after a checkpoint. The cancelled request drained;
  its follow-up restored 1,664 tokens and returned the expected answer, with
  no failed requests.

The same candidate binary was tested in OFF / ON / ON / OFF order, restarting
and warming the server each time. Each round used an uncached 4,072-token prompt,
a 1,171-token answer copying an explicit integer array, then a 5,276-token
follow-up retrieving three codes from the prompt. Temperature was zero and
there were no other submitted requests. All answer and follow-up bytes matched
across the four rounds; draft proposal and acceptance counts also matched.

| Mode | Long answer (s) | Follow-up (s) | Pair (s) | Follow-up cached tokens |
| --- | ---: | ---: | ---: | ---: |
| OFF, round 1 | 38.712 | 7.879 | 46.591 | 4,064 |
| ON, round 2 | 38.745 | 1.591 | 40.336 | 5,184 |
| ON, round 3 | 38.737 | 1.590 | 40.327 | 5,184 |
| OFF, round 4 | 38.710 | 7.887 | 46.596 | 4,064 |

Mean pair latency fell from 46.594 to 40.331 seconds (13.4%); follow-up latency
fell 79.8%. The long answer added 0.030 seconds (0.08%), too small for a broad
throughput claim from these four rounds. Each enabled long answer published
nine generated checkpoints. This workload demonstrates reuse of a long answer;
short outputs or decode cycles that miss aligned page boundaries may benefit
less or not at all.

The near-limit run used a 99,481-token cold prompt and produced the same
1,171-token answer byte for byte in 897.359 seconds (873.254 seconds prefill,
24.094 seconds decode). Under host memory pressure, the governor reclaimed
the cached states before the follow-up. That follow-up therefore re-prefilled
its 100,685-token prompt, taking 889.175 seconds with zero cached tokens. All
three retrieved codes were correct, and the answer bytes matched the short
fixture. Across both requests, 5-second samples recorded peaks of 23.945 GiB
governed memory and 23.518 GiB native physical footprint, with host available
memory as low as 1.862 GiB. Pressure reached `warning`; there were 37 refused
reservations but no request failures. Swap stayed at 243.88 MiB. ANE remained
active without reruns. This is a limitation of caching at near-capacity host
pressure: the 24 GiB budget permits the request, but does not promise that its
reusable states remain resident. The 4K measurements above must not be read as
a near-limit speedup claim.

The final build (`src-6bebb4632dad68b23e48d32c05e551939b988e94c3eeaac2ec1ff037ebcd947b`)
was deployed through the existing LaunchAgent on port 1234. Its 4K production
smoke test published nine generated checkpoints and restored 5,184 tokens for
the correct follow-up in 1.591 seconds. BF16 KV/state, 24 GiB, ANE split prefill,
the 102,265-token plan and ten-minute idle release remain configured. A smaller
1,023-token prompt with a 531-token answer also returned correct results but
published no generated checkpoint, illustrating the natural-boundary limitation.

After validation, removed approximately 317 MiB of duplicate candidate/CPU
builds, raw test requests and responses, temporary scripts and logs, unused ANE
calibration programs/records, shader-validation cache and Python bytecode.
The deployed build and its incremental inputs, current ANE program/calibration,
model files, Python environment and active service records remain. This is the
sum of allocated file sizes, not an APFS physical-space measurement. The probe,
regression tests and numerical results are retained in Git.
