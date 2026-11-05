"""Deploy only an eligible local factorial candidate, with automatic rollback."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/m4-local-combined-20261106'
sys.path.insert(0, str(ROOT / 'build/m4-generation-cache'))
from service import BASELINE, SCRIPT, install, model, request, manifest as previous
from quality import check as check_first, module as original_quality

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(16 * 2**20):
            h.update(chunk)
    return h.hexdigest()

decision = json.loads((OUT / 'decision.json').read_text())
selected = decision['selected']
assert selected and decision['variants'][selected]['eligible']
assert json.loads((OUT / 'validation.json').read_text())['result'] == 'PASS'
candidate = json.loads((OUT / 'candidate.json').read_text())
spec = json.loads((OUT / 'spec.json').read_text())
assert BASELINE.encode() == (OUT / 'rollback/serve-q6.sh').read_bytes()
before = request('/status')
assert before['ready'] and not before['active_requests']
assert before['identity']['cache']['build_id'] == spec['baseline_build_id']
deployment = OUT / 'deployment'
deployment.mkdir(exist_ok=False)
save(deployment / 'before-status.json', before)
package = Path('/Users/cofiavcin/splash_server_config') / ('local-m4-' + selected + '-' + candidate['build_id'][4:16])
engine = package / 'engine'
engine.mkdir(parents=True, exist_ok=False)
for name, expected in candidate['files'].items():
    source = Path(candidate['engine']) / name
    assert digest(source) == expected
    shutil.copy2(source, engine / name)
    assert digest(engine / name) == expected
weights = json.loads((Path(previous['package']) / 'model-integrity.json').read_text())
for name, info in weights.items():
    assert Path(info['path']).stat().st_size == info['bytes']
    assert digest(Path(info['path'])) == info['sha256']
    print('ORIGINAL_WEIGHT_SHA_PASS', name, flush=True)
save(package / 'model-integrity.json', weights)
(package / 'serve-q6.rollback.sh').write_text(BASELINE)
shutil.copy2(ROOT / 'build/engine/build-identity.json', package / 'build-identity.json')
patch = subprocess.check_output(['git', 'diff', '--binary', 'main', '--', 'runtime', 'dev/tests/engine', 'dev/benchmarks', 'LOCAL_OPTIMIZATION.txt'], cwd=ROOT)
for name in subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines():
    if name.startswith(('runtime/', 'dev/tests/engine/', 'dev/benchmarks/')):
        result = subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', name], cwd=ROOT, capture_output=True)
        assert result.returncode == 1
        patch += result.stdout
(package / 'source.patch').write_bytes(patch)
validation = package / 'validation'
validation.mkdir()
for path in OUT.glob('*.log'):
    shutil.copy2(path, validation / path.name)
for name in ['validation.json', 'comparison.json', 'decision.json', 'spec.json', 'serving-complete.json', 'serving-started.json']:
    shutil.copy2(OUT / name, validation / name)
options = spec['variants'][selected]
configured = re.sub(r'--binary "[^"]+"', f'--binary "{engine}/splash"', BASELINE, count=1)
exports = (f'export SPLASH_LOCAL_PARALLEL_GATEUP={int(options["joint"])}\n'
           f'export SPLASH_LOCAL_ATTENTION_FLAGS={int(options["joint"])}\n'
           f'export SPLASH_LOCAL_KEEP_MESSAGE_OFFLOAD={int(options["offload"])}\n'
           'unset SPLASH_LOCAL_OPTIMIZATION_TRACE\n')
configured = configured.replace('export PYTHONDONTWRITEBYTECODE', exports + 'export PYTHONDONTWRITEBYTECODE', 1)
manifest = {'build_id': candidate['build_id'], 'baseline_build_id': previous['build_id'],
    'package': str(package), 'variant': selected, 'files': candidate['files'],
    'model_integrity': weights, 'memory_limit_bytes': 44 * 2**30, 'kv_format': 'bf16',
    'state_format': 'fp32', 'local_only': True, 'source_baseline_commit': 'f2e25f9b',
    'production_script_sha256': hashlib.sha256(configured.encode()).hexdigest(),
    'experiment': str(OUT), 'smoke_passed': False}

source = ROOT / 'build/m4-generation-cache/serve_compare_final.py'
tree = ast.parse(source.read_text())
selected_nodes = [n for n in tree.body if
    (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cases' for t in n.targets)) or
    (isinstance(n, ast.FunctionDef) and n.name == 'check_second')]
exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(source), 'exec'), globals())

def verify(status):
    assert status['ready'] and status['metal']['healthy'] and not status['active_requests']
    assert status['identity']['cache']['build_id'] == candidate['build_id']
    assert status['identity']['kv'] == before['identity']['kv']
    assert status['memory_plan']['model']['state_format'] == 'fp32'
    budget = status['memory_plan']['budget']
    assert budget['configured_memory_limit_bytes'] == budget['hard_budget_bytes'] == 44 * 2**30
    assert status['memory_governor']['limit_bytes'] == 44 * 2**30
    assert status['maximum_context_tokens'] == before['maximum_context_tokens']
    assert status['disk']['limit_bytes'] == 32 * 2**30
    for name in ['share', 'minimum_rows']:
        assert status['ane_ffn'][name] == before['ane_ffn'][name]

try:
    model('unload')
    install(configured)
    model('load')
    verify(request('/status'))
    case, effort, path, followup, oracle = next(c for c in cases if c[0] == 'ttl-local')
    payload = json.loads(path.read_text())
    payload.update(max_tokens=8192, temperature=1, top_k=20, top_p=.95, seed=42, reasoning_effort=effort)
    def call(payload, stage):
        prior = request('/status')
        started = time.monotonic()
        response = request('/v1/chat/completions', payload)
        wall = time.monotonic() - started
        after = request('/status')
        record = {'wall_s': wall, 'response': response, 'before': prior, 'after': after}
        save(deployment / (stage + '.json'), record)
        verify(after)
        assert after['requests']['failed'] == prior['requests']['failed']
        assert after['requests']['submitted'] - prior['requests']['submitted'] == 1
        assert response['choices'][0]['finish_reason'] != 'length'
        return record
    one = call(payload, 'first')
    directory = deployment / 'first-artifact'
    q1 = check_first(case, one['response'], directory)
    assert q1['pass'], q1
    follow = copy.deepcopy(payload)
    message = one['response']['choices'][0]['message']
    follow['messages'].append(message)
    for tool in message.get('tool_calls', []):
        follow['messages'].append({'role': 'tool', 'tool_call_id': tool['id'],
            'content': 'The requested file modification was applied successfully.'})
    follow['messages'].append({'role': 'user', 'content': followup})
    two = call(follow, 'second')
    q2 = check_second(case, directory, two['response'], deployment / 'second-artifact', oracle)
    assert q2['pass'], q2
    assert two['response']['timings']['cache_n'] >= one['response']['usage']['prompt_tokens'] + 32
    after = request('/status')
    verify(after)
    save(deployment / 'verified-status.json', after)
    save(deployment / 'production-quality.json', [q1, q2])
    (package / 'serve-q6.sh').write_text(configured)
    manifest['smoke_passed'] = True
    save(package / 'manifest.json', manifest)
    save(deployment / 'manifest.json', manifest)
    # Future local experiment guards must restore this deployment.
    guard = ROOT / 'build/m4-generation-cache/service.py'
    old = guard.read_text()
    needle = 'known_manifests=[p for p in [\n'
    assert old.count(needle) == 1
    guard.write_text(old.replace(needle, needle + "    ROOT/'build/m4-local-combined-20261106/deployment/manifest.json',\n", 1))
    current = Path('/Users/cofiavcin/splash_server_config/CURRENT.txt')
    shutil.copy2(current, package / 'CURRENT.before.txt')
    prefix = ('本机定向优化部署（2026-11-06）：local/m4max-q6k；引擎 ' + candidate['build_id'] +
        '，方案 ' + manifest['variant'] + '。原 Q6_K / DFlash、BF16 未量化 KV、FP32 状态、44 GiB、32 GiB 内盘持久缓存。'
        '此前四项本地优化保留；新四组对照、完整响应和质量限制见 ' + str(OUT) + '。'
        '启动脚本 serve-q6.sh 已指向版本化本地包；回滚脚本见 ' + str(package / 'serve-q6.rollback.sh') + '。无远端操作。\n\n')
    current.write_text(prefix + current.read_text())
    print('LOCAL_COMBINED_DEPLOYMENT_COMPLETE', selected, candidate['build_id'], flush=True)
except BaseException:
    try:
        model('unload')
    finally:
        install(BASELINE)
        model('load')
    raise
