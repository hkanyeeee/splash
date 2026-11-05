"""Local maintenance-only factorial serving experiment; retain every response.

Requires the frozen local fixtures/seeds and validation artifacts under build/.
No model files, production caches, or historical experiment caches are removed.
"""
from pathlib import Path
import ast
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/m4-local-combined-20261106'
sys.path.insert(0, str(ROOT / 'build/m4-generation-cache'))
from service import BASELINE, released, request, model, install, manifest as baseline
from quality import check as check_first, module as original_quality
sys.path.insert(0, str(ROOT / 'build/m4-keep-message-offload'))
from completion import inspect as inspect_completion

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(16 * 2**20):
            h.update(block)
    return h.hexdigest()

def metadata(path):
    return {str(p.relative_to(path)): {'bytes': p.stat().st_size,
            'mtime_ns': p.stat().st_mtime_ns, 'inode': p.stat().st_ino}
            for p in sorted(path.rglob('*')) if p.is_file()}

SPEC = json.loads((OUT / 'spec.json').read_text())
CANDIDATE = json.loads((OUT / 'candidate.json').read_text())
BASE_STATUS = json.loads((OUT / 'rollback/status.json').read_text())
assert json.loads((OUT / 'validation.json').read_text())['result'] == 'PASS'
assert baseline['build_id'] == SPEC['baseline_build_id']
assert BASELINE.encode() == (OUT / 'rollback/serve-q6.sh').read_bytes()
assert not (OUT / 'serving-started.json').exists()
for name, expected in CANDIDATE['files'].items():
    assert digest(Path(CANDIDATE['engine']) / name) == expected
SOURCE = ROOT / 'build/m4-generation-cache/serve_compare_final.py'
tree = ast.parse(SOURCE.read_text())
selected = [n for n in tree.body if
    (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cases' for t in n.targets)) or
    (isinstance(n, ast.FunctionDef) and n.name == 'check_second')]
assert len(selected) == 2
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), globals())
SHORT_CASES = cases
NAMESPACE = 'eae4fe802baee1f14d2841123abd5e07'
SERVER_LOG = Path('/Users/cofiavcin/model_gateway/logs/splash.log')
ROWS = []
SEEDS = {64: ROOT / 'build/m4-tail64/cache-staged64' / NAMESPACE,
         192: ROOT / 'build/m4-serving192/cache-seed' / NAMESPACE}
SEED_METADATA = {str(n): metadata(p) for n, p in SEEDS.items()}
assert all(p.stat().st_dev == ROOT.stat().st_dev for p in SEEDS.values())
save(OUT / 'serving-started.json', {
    'started_at': time.time(), 'spec_sha256': digest(OUT / 'spec.json'),
    'runner_sha256': digest(Path(__file__)), 'candidate': CANDIDATE,
    'report_sha256': digest(ROOT / 'dev/benchmarks/local_m4_combined_report.py'),
    'deploy_sha256': digest(ROOT / 'dev/benchmarks/local_m4_combined_deploy.py'),
    'baseline': baseline, 'seeds': SEED_METADATA})

def terminate(signum, frame):
    raise RuntimeError(f'signal {signum}; restore accepted deployment')
signal.signal(signal.SIGTERM, terminate)

def check_status(status, variant):
    expected = baseline['build_id'] if variant == 'baseline' else CANDIDATE['build_id']
    assert status['ready'] and status['metal']['healthy']
    assert status['identity']['cache']['build_id'] == expected
    assert status['identity']['kv']['format'] == 'bf16'
    assert status['identity']['kv']['quantization'] == 'none'
    assert status['identity']['kv'] == BASE_STATUS['identity']['kv']
    assert status['memory_plan']['model']['state_format'] == 'fp32'
    assert status['maximum_context_tokens'] == BASE_STATUS['maximum_context_tokens']
    assert status['ane_ffn']['share'] == BASE_STATUS['ane_ffn']['share']
    assert status['ane_ffn']['minimum_rows'] == BASE_STATUS['ane_ffn']['minimum_rows']
    b = status['memory_plan']['budget']
    assert b['configured_memory_limit_bytes'] == b['hard_budget_bytes'] == 44 * 2**30
    assert status['memory_governor']['limit_bytes'] == 44 * 2**30
    assert status['disk']['capacity_bytes'] == status['disk']['limit_bytes'] == 32 * 2**30
    assert status['memory_actual']['allocated_bytes'] <= 44 * 2**30

def call(payload, context, phase, variant, workload, index, case, stage, folder):
    key = f'{phase}-{variant}-{workload}-{index}-{case}-{stage}'
    before = request('/status')
    check_status(before, variant)
    assert not before['active_requests']
    log_start = SERVER_LOG.stat().st_size
    started = time.monotonic()
    response = request('/v1/chat/completions', payload)
    response_end = time.monotonic()
    after = request('/status')
    def read_log():
        with SERVER_LOG.open('rb') as stream:
            stream.seek(log_start)
            return stream.read().decode(errors='replace')
    log = read_log()
    completion = inspect_completion(log)
    maintenance = 0.0
    if completion['new_writes'] and not completion['finished_before_native_done']:
        deadline = response_end + 30
        while completion['unfinished'] and not completion['failed'] and time.monotonic() < deadline:
            time.sleep(.02)
            after = request('/status')
            log = read_log()
            completion = inspect_completion(log)
        maintenance = time.monotonic() - response_end
    (folder / (key + '-engine.log')).write_text(log)
    engagement = {'pair': len(re.findall(r'LOCAL_PARALLEL_GATEUP ', log)),
                  'flags': len(re.findall(r'LOCAL_ATTENTION_FLAGS phase=verify', log)),
                  'retentions': [dict(zip(['request', 'tokens', 'block', 'write_started'], map(int, m)))
                     for m in re.findall(r'LOCAL_KEEP_MESSAGE_OFFLOAD request=(\d+) tokens=(\d+) block=(\d+) write_started=([01])', log)]}
    record = {'key': key, 'context': context, 'phase': phase, 'variant': variant,
              'workload': workload, 'index': index, 'case': case, 'stage': stage,
              'wall_s': response_end - started + maintenance,
              'response_wall_s': response_end - started, 'maintenance_wall_s': maintenance,
              'response': response, 'before': before, 'after': after,
              'other_submissions': after['requests']['submitted'] - before['requests']['submitted'] - 1,
              'engagement': engagement, 'write_completion': completion}
    save(folder / (key + '.json'), record)
    check_status(after, variant)
    assert record['other_submissions'] == 0
    assert not completion['unfinished'] and not completion['failed']
    assert after['requests']['failed'] == before['requests']['failed']
    assert response['choices'][0]['finish_reason'] != 'length'
    assert bool(engagement['pair']) == SPEC['variants'][variant]['joint']
    assert bool(engagement['flags']) == SPEC['variants'][variant]['joint']
    if not SPEC['variants'][variant]['offload']:
        assert not engagement['retentions']
    if index == 0 and stage != 'second':
        assert response['timings']['cache_n'] >= (60000 if context == 64 else 187000)
    return record, folder / (key + '-artifact')

def append(record, quality):
    item = {k: record[k] for k in ['key', 'context', 'phase', 'variant', 'workload',
        'index', 'case', 'stage', 'wall_s', 'response_wall_s', 'maintenance_wall_s',
        'other_submissions', 'engagement', 'write_completion']}
    item.update(quality=quality, timings=record['response']['timings'], usage=record['response']['usage'],
        memory_pressure=record['after']['memory_pressure'],
        system_pressure=record['after']['memory_governor']['system_pressure'],
        thermal_state=record['after']['thermal_state'],
        allocated_bytes=record['after']['memory_actual']['allocated_bytes'],
        checkpoint_bytes=record['after']['state']['checkpoint_bytes'])
    ROWS.append(item)
    save(OUT / 'summary.json', ROWS)
    save(OUT / 'run-state.json', {'updated_at': time.time(), 'complete_requests': len(ROWS),
        'expected_requests': 256, 'last': item})
    print('RESPONSE', len(ROWS), context, record['key'], 'wall', round(record['wall_s'], 3),
          'quality', quality['pass'], 'computed', item['timings']['prompt_n']-item['timings']['cache_n'],
          'retentions', len(item['engagement']['retentions']), flush=True)

def retire_scratch(cache, cohort, phase):
    assert cache == cohort / f'cache-{phase}' and cache.is_dir() and not cache.is_symlink()
    assert OUT in cache.parents and len([r for r in ROWS if r['context'] == int(cohort.name) and r['phase'] == phase]) == 16
    processes = subprocess.check_output(['ps', '-axo', 'pid,command'], text=True)
    assert not any(str(cache) in line for line in processes.splitlines() if ' serve-native ' in line)
    files = sorted(p for p in cache.rglob('*') if p.is_file())
    assert not any(p.is_symlink() for p in cache.rglob('*'))
    index = cohort / f'cache-{phase}-index'
    index.mkdir(exist_ok=False)
    inventory = {}
    with (cache / NAMESPACE / 'lock').open('rb') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for p in files:
            inventory[str(p.relative_to(cache))] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
            if p.name.endswith('.records'):
                shutil.copy2(p, index / p.name)
        save(index / 'manifest.json', {'scope': 'New diagnostic scratch only; full hash/index retained before removal after unload.',
             'files': inventory, 'completed_at': time.time()})
    shutil.rmtree(cache)
    print('SCRATCH_RETIRED', cohort.name, phase, 'internal_free', shutil.disk_usage(ROOT).free, flush=True)

with released():
    for context in SPEC['contexts']:
        cohort = OUT / str(context)
        cohort.mkdir(exist_ok=False)
        folder = cohort / 'serving'
        folder.mkdir()
        if context == 64:
            forks = json.loads((ROOT / 'build/m4-tail-checkpoint/fixture-manifest.json').read_text())
            workflows = SHORT_CASES
        else:
            long = json.loads((ROOT / 'build/m4-serving192/fixture-manifest.json').read_text())
            forks = [r for r in long['rows'] if r['workload'] == 'forks']
            workflows = [(r['case'], r['effort'], Path(r['path']), r['followup'], r['oracle']) for r in long['followups']]
        for phase, variant in enumerate(SPEC['phase_order']):
            assert shutil.disk_usage(ROOT).free > 42 * 2**30
            cache = cohort / f'cache-{phase}'
            cache.mkdir(exist_ok=False)
            subprocess.run(['cp', '-cR', str(SEEDS[context]), str(cache / NAMESPACE)], check=True)
            binary = Path(baseline['package']) / 'engine/splash' if variant == 'baseline' else Path(CANDIDATE['engine']) / 'splash'
            config = re.sub(r'--binary "[^"]+"', f'--binary "{binary}"', BASELINE, count=1)
            options = SPEC['variants'][variant]
            exports = (f'export SPLASH_LOCAL_PARALLEL_GATEUP={int(options["joint"])}\n'
                       f'export SPLASH_LOCAL_ATTENTION_FLAGS={int(options["joint"])}\n'
                       f'export SPLASH_LOCAL_KEEP_MESSAGE_OFFLOAD={int(options["offload"])}\n'
                       'export SPLASH_LOCAL_OPTIMIZATION_TRACE=1\n')
            config = config.replace('export PYTHONDONTWRITEBYTECODE', exports + 'export PYTHONDONTWRITEBYTECODE', 1)
            config = config.replace('"$BASE/cache/q6-1.3.2"', f'"{cache}"')
            install(config)
            print('COHORT_START', context, phase, variant, flush=True)
            model('load')
            status = request('/status')
            check_status(status, variant)
            save(folder / f'{phase}-{variant}-loaded.json', status)
            for index, row in enumerate(forks if phase < 4 else list(reversed(forks))):
                path = ROOT / 'build/m4-tail-checkpoint/fixtures' / (row['case'] + '.json') if context == 64 else Path(row['path'])
                assert digest(path) == row['sha256']
                payload = json.loads(path.read_text())
                record, artifact = call(payload, context, phase, variant, 'forks', index, row['case'], 'single', folder)
                assert record['response']['usage']['prompt_tokens'] == row['prompt_tokens']
                append(record, check_first(row['case'], record['response'], artifact))
            for index, (case, effort, path, followup, oracle) in enumerate(workflows if phase < 4 else list(reversed(workflows))):
                first = json.loads(path.read_text())
                first.update(max_tokens=8192, temperature=1, top_k=20, top_p=.95, seed=42, reasoning_effort=effort)
                one, directory = call(first, context, phase, variant, 'sequential', index, case, 'first', folder)
                append(one, check_first(case, one['response'], directory))
                second = copy.deepcopy(first)
                message = one['response']['choices'][0]['message']
                second['messages'].append(message)
                for tool in message.get('tool_calls', []):
                    second['messages'].append({'role': 'tool', 'tool_call_id': tool['id'],
                        'content': 'The requested file modification was applied successfully.'})
                second['messages'].append({'role': 'user', 'content': followup})
                two, artifact = call(second, context, phase, variant, 'sequential', index, case, 'second', folder)
                append(two, check_second(case, directory, two['response'], artifact, oracle))
            # Settle normal delayed persistence before stopping this cohort.
            deadline = time.monotonic() + 35
            while True:
                status = request('/status')
                assert not status['active_requests']
                if not status['disk']['write_behind']['waiting'] and not status['disk']['kv_pending_pages']:
                    break
                assert time.monotonic() < deadline, 'persistence did not settle'
                time.sleep(.1)
            save(folder / f'{phase}-{variant}-settled.json', status)
            model('unload')
            retire_scratch(cache, cohort, phase)
for n, p in SEEDS.items():
    assert metadata(p) == SEED_METADATA[str(n)]
save(OUT / 'serving-complete.json', {'result': 'COMPLETE', 'requests': len(ROWS),
    'seed_metadata_unchanged': True, 'finished_at': time.time(),
    'restored_build_id': request('/status')['identity']['cache']['build_id']})
assert len(ROWS) == 256
print('LOCAL_COMBINED_FACTORIAL_COMPLETE_AND_RESTORED', flush=True)
