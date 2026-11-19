#!/bin/sh
# Isolated 98 candidate. --check prepares local metadata and parses both CLIs.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
validation=/Users/hewenxin/Library/Caches/splash-170-m2-98/evidence/validation
python=/Users/hewenxin/code/splash/.venv/bin/python
weights=/Users/hewenxin/gguf_models
case ${1:---check} in
    --check|--serve) mode=${1:---check} ;;
    *) echo 'usage: run-qwen36-validation.sh [--check|--serve]' >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || exit 2
export PYTHONPATH="$root" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export TMPDIR="$validation/tmp" SPLASH_ISOLATED_STATE_DIR="$validation/state"
export XDG_CACHE_HOME="$validation/xdg" HF_HOME="$validation/hf"
export SPLASH_RUNTIME_LANES=2 SPLASH_PREFILL_ROWS=512
mkdir -p "$TMPDIR" "$SPLASH_ISOLATED_STATE_DIR"
prepared=$("$python" -B -P -m install.local \
    --models "$validation/models" \
    --draft-model "$weights/drafts/Qwen3.6-35B-A3B-DFlash2/51ef7b6923ad6c14cb1bb41c37a9041446496ab6" \
    --mmproj "$weights/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf" \
    "$weights/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
set -- "$prepared" --model local/Qwen3.6-35B-A3B-UD-Q4_K_M --binary "$root/build/splash" \
    --served-model-name qwen3.6-35b-a3b --announce-served-name \
    --host 127.0.0.1 --port 32235 --max-memory 26G --max-context 262144 \
    --kv-format bf16 --state-format bf16 --cached-states exact --activations bf16 --verify-rows all \
    --queue-size 2 --idle-release off --max-cache-disk 32G --persistent-cache \
    --cache-dir "$validation/prefix-cache"
if [ "$mode" = --check ]; then
    exec "$python" -B -P -c '
import json, sys
from server.server import _native_command, parse_args
args = parse_args(sys.argv[1:])
print(json.dumps({"server": sys.argv[1:], "native": _native_command(args, args.model_root)}, indent=2))
' "$@"
fi
exec "$python" -u -B -P -m server.server "$@"
