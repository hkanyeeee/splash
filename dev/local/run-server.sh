#!/bin/sh
# Local M2 Max / 32 GiB deployment. Overrides are for isolated benchmarks.
set -eu
checkout=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
# Passed through the source launcher's make invocation as well as manual builds.
export SPLASH_RUNTIME_LANES=1
export SPLASH_PREFILL_ROWS="${SPLASH_PREFILL_ROWS:-512}"
case "${SPLASH_LOCAL_ANE:-auto}" in
  off) set -- --disable-ane "$@" ;;
  auto) ;;
  *) echo "SPLASH_LOCAL_ANE must be off or auto" >&2; exit 2 ;;
esac
exec "$checkout/splash" serve \
  --model local/Qwen3.8-27B-UD-Q4_K_M \
  --model-dir /Users/hewenxin/gguf_models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --served-model-name qwen3.8-27b-mtp-98 \
  --served-model-name qwen3.8-27b \
  --announce-served-name \
  --offline --language-only \
  --host "${SPLASH_LOCAL_HOST:-0.0.0.0}" \
  --port "${SPLASH_LOCAL_PORT:-1234}" \
  --max-memory "${SPLASH_LOCAL_MEMORY:-24G}" \
  --max-context "${SPLASH_LOCAL_CONTEXT:-auto}" \
  --kv-format bf16 --state-format bf16 \
  --queue-size 1 \
  --max-cache-disk 0 \
  --idle-release "${SPLASH_LOCAL_IDLE:-10m}" \
  --default-reasoning-effort medium \
  "$@"
