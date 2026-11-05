"""Build and validate the formal combined implementation before serving."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/m4-local-combined-20261106'
sys.path.insert(0, str(ROOT / 'build/m4-generation-cache'))
from service import released

def run(args, name, env=None):
    print('START', name, flush=True)
    with (OUT / (name + '.log')).open('w') as log:
        subprocess.run(list(map(str, args)), cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print('PASS', name, flush=True)

assert not (OUT / 'validation.json').exists()
resume = '--resume' in sys.argv
flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror',
         '-Wno-unused-function', '-Iruntime', '-Idev', '-Idev/tests/engine',
         '-mmacosx-version-min=26.4']
frameworks = ['-framework', 'Foundation', '-framework', 'Metal',
              '-framework', 'IOKit', '-framework', 'IOSurface']
if not resume:
    run(['clang++', *flags, 'dev/tests/engine/local_message_offload_test.cpp',
        'build/engine/libsplash.a', *frameworks, '-o', OUT / 'cache-probe'], 'compile-cache-probe')
for name in ([] if resume else ['engine', 'engine-memory-plan', 'memory-audit', 'memory-governor', 'runtime-status']):
    args = [ROOT / 'build/engine-tests' / name]
    if name == 'runtime-status':
        args.append('dev/tests/engine/status_golden.json')
    run(args, 'cpu-' + name)
if not resume:
    run([OUT / 'cache-probe'], 'cpu-message-offload')
    run([ROOT / 'build/engine-tests/linear-plan', '--cpu'], 'cpu-linear-plan')
old = ROOT / 'build/m4-joint-exact'
for name in ([] if resume else ['oracle.mm', 'pair.hpp', 'attention.mm', 'screen.mm']):
    shutil.copy2(old / name, OUT / name)
if not resume:
    run(['make', 'build/engine/model/RuntimeInstrumented.o'], 'build-instrumented-runtime')
for name, extra in ([] if resume else [('oracle', []), ('attention', []),
                    ('screen', ['build/engine/model/RuntimeInstrumented.o'])]):
    run(['clang++', *flags, '-fobjc-arc', OUT / (name + '.mm'),
         *extra, 'build/engine/libsplash.a', *frameworks, '-o', OUT / name],
        'compile-' + name)
engine = OUT / 'engine'
if not resume:
    engine.mkdir(exist_ok=False)
    for name in ['splash', 'splash.metallib', 'host-step.metallib']:
        shutil.copy2(ROOT / 'build' / name, engine / name)
identity = json.loads((ROOT / 'build/engine/build-identity.json').read_text())
(OUT / 'candidate.json').write_text(json.dumps({
    **identity, 'engine': str(engine),
    'files': {name: hashlib.sha256((engine / name).read_bytes()).hexdigest()
              for name in ['splash', 'splash.metallib', 'host-step.metallib']}}, indent=2) + '\n')
env = {**os.environ, 'MTL_SHADER_VALIDATION': '1'}
with released():
    if not resume:
        run([ROOT / 'build/engine-tests/linear-plan', engine / 'splash.metallib'],
            'gpu-linear-plan', env)
        run([ROOT / 'build/engine-tests/draft-attention', engine / 'splash.metallib'],
            'gpu-draft-attention', env)
        run([OUT / 'oracle', engine / 'splash.metallib'], 'gpu-pair-oracle', env)
    else:
        assert 'BORROW_PAIR_ORACLE_COMPLETE checks=600' in (OUT / 'gpu-pair-oracle.log').read_text()
        assert 'Linear plans: PASS' in (OUT / 'gpu-linear-plan.log').read_text()
        assert 'draft_attention_metal_test: PASS' in (OUT / 'gpu-draft-attention.log').read_text()
    run([OUT / 'attention', engine / 'splash.metallib', 'validate'], 'gpu-attention', env)
    plain = {**os.environ, 'SPLASH_LOCAL_GENERATION_CACHE': '1'}
    plain.pop('MTL_SHADER_VALIDATION', None)
    run([OUT / 'oracle', engine / 'splash.metallib'], 'gpu-pair-plain', plain)
    run([OUT / 'attention', engine / 'splash.metallib', 'timing'], 'gpu-attention-plain', plain)
    spec = json.loads((old / 'spec.json').read_text())
    fixture = spec['histories'][0]
    assert hashlib.sha256(Path(fixture['tokens']).read_bytes()).hexdigest() == fixture['sha256']
    run([OUT / 'screen', engine / 'splash.metallib', spec['model_root'],
         fixture['tokens'], fixture['history']], 'native-64000', plain)
files = [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.log', '.py', '.json', '.mm', '.hpp']]
(OUT / 'validation.json').write_text(json.dumps({
    'result': 'PASS', 'candidate_build_id': identity['build_id'],
    'scope': 'Formal cache lifecycle, existing CPU suites, GPU planner/draft windows, full-format FP64 pair oracle, all-visible-key attention controls, full-model64K forced-token raw-logit/GDN bit comparisons. New full-model188K proof is through complete serving, with prior unchanged shader lineage retained.',
    'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}, indent=2) + '\n')
print('LOCAL_COMBINED_VALIDATION_COMPLETE', flush=True)
