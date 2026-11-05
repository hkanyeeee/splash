"""Report every factorial result; cache strata are post-hoc diagnostics only."""
from pathlib import Path
import collections
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/m4-local-combined-20261106'
SPEC = json.loads((OUT / 'spec.json').read_text())
assert json.loads((OUT / 'serving-complete.json').read_text())['requests'] == 256
rows = json.loads((OUT / 'summary.json').read_text())
assert len(rows) == 256 and all(r['other_submissions'] == 0 for r in rows)
records = {}
files = {}
for row in rows:
    path = OUT / str(row['context']) / 'serving' / (row['key'] + '.json')
    raw = path.read_bytes()
    files[str(path)] = hashlib.sha256(raw).hexdigest()
    record = json.loads(raw)
    records[row['context'], row['key']] = record
    assert record['wall_s'] == row['wall_s']
    assert record['response']['timings'] == row['timings']
    assert record['response']['choices'][0]['finish_reason'] != 'length'

def aggregate(items):
    walls = sorted(r['wall_s'] for r in items)
    return {'requests': len(items), 'quality_passes': sum(r['quality']['pass'] for r in items),
        'wall_s': math.fsum(r['wall_s'] for r in items),
        'response_wall_s': math.fsum(r['response_wall_s'] for r in items),
        'maintenance_wall_s': math.fsum(r['maintenance_wall_s'] for r in items),
        'maximum_request_wall_s': walls[-1] if walls else 0,
        'p95_request_wall_s': walls[math.ceil(.95 * len(walls)) - 1] if walls else 0,
        'prefill_s': math.fsum(r['timings']['prompt_ms'] / 1000 for r in items),
        'decode_s': math.fsum(r['timings']['predicted_ms'] / 1000 for r in items),
        'computed_tokens': sum(r['timings']['prompt_n']-r['timings']['cache_n'] for r in items),
        'output_tokens': sum(r['usage']['completion_tokens'] for r in items),
        'retentions': sum(len(r['engagement']['retentions']) for r in items),
        'new_backup_writes': sum(r['write_completion']['new_writes'] for r in items),
        'memory_warning_requests': sum(r['memory_pressure'] != 'normal' for r in items),
        'system_pressure_requests': sum(r['system_pressure'] != 'normal' for r in items),
        'thermal_warning_requests': sum(r['thermal_state'] != 'nominal' for r in items),
        'maximum_allocated_bytes': max((r['allocated_bytes'] for r in items), default=0)}

def compare(a, b):
    a, b = aggregate(a), aggregate(b)
    return {'baseline': a, 'candidate': b,
        'wall_speedup_pct': 100 * (a['wall_s'] / b['wall_s'] - 1),
        'wall_reduction_pct': 100 * (1 - b['wall_s'] / a['wall_s'])}

def canonical(row):
    response = records[row['context'], row['key']]['response']
    choice = response['choices'][0]
    message = choice['message']
    return (message.get('content'), message.get('reasoning_content'),
        [t['function'] for t in message.get('tool_calls', [])], choice['finish_reason'])

def failure_deltas(items):
    violations = []
    for row in items:
        rec = records[row['context'], row['key']]
        for group, names in {'requests': ['failed'], 'state': ['offload_failures'],
            'cache': ['lost_state_misses', 'replay_state_publication_failures',
                      'checkpoint_publication_failures', 'junction_materialization_failures'],
            'disk': ['kv_demotion_failures', 'kv_restore_failures', 'kv_copy_failures'],
            'metrics': ['metal_failures', 'capacity_failures']}.items():
            for name in names:
                delta = rec['after'][group].get(name, 0) - rec['before'][group].get(name, 0)
                if delta:
                    violations.append({'key': row['key'], 'group': group, 'counter': name, 'delta': delta})
    return violations

reports = {}
for context in SPEC['contexts']:
    cohort = [r for r in rows if r['context'] == context]
    base = [r for r in cohort if r['variant'] == 'baseline']
    reports[str(context)] = {}
    for variant in ['joint', 'offload', 'combined']:
        candidate = [r for r in cohort if r['variant'] == variant]
        pairs, identical_a, identical_b = [], [], []
        strata = collections.defaultdict(lambda: ([], []))
        for reverse in [False, True]:
            for workload in ['forks', 'sequential']:
                for case in sorted({r['case'] for r in base if r['workload'] == workload}):
                    a = sorted([r for r in base if (r['phase'] >= 4) == reverse and r['workload'] == workload and r['case'] == case], key=lambda r: r['stage'])
                    b = sorted([r for r in candidate if (r['phase'] >= 4) == reverse and r['workload'] == workload and r['case'] == case], key=lambda r: r['stage'])
                    assert len(a) == len(b) == (1 if workload == 'forks' else 2)
                    same = [canonical(r) for r in a] == [canonical(r) for r in b]
                    cache_equal = all((x['timings']['prompt_n'], x['timings']['cache_n']) ==
                        (y['timings']['prompt_n'], y['timings']['cache_n']) for x, y in zip(a, b))
                    pairs.append({'reverse': reverse, 'workload': workload, 'case': case,
                        'all_turn_outputs_identical': same, 'per_turn_cache_counts_equal': cache_equal,
                        'additional_behavior_failures': sum(x['quality']['pass'] and not y['quality']['pass'] for x, y in zip(a, b)),
                        **compare(a, b)})
                    key = f'outputs_equal={same},cache_counts_equal={cache_equal}'
                    strata[key][0].extend(a)
                    strata[key][1].extend(b)
                    if same:
                        identical_a.extend(a)
                        identical_b.extend(b)
        same_count = sum(p['all_turn_outputs_identical'] for p in pairs)
        identical = compare(identical_a, identical_b) if same_count else None
        reports[str(context)][variant] = {'overall': compare(base, candidate),
            'identical_complete_workflows': same_count, 'compared_workflows': len(pairs),
            'identical_output_subset': identical, 'workflow_pairs': pairs,
            'additional_behavior_failures': sum(p['additional_behavior_failures'] for p in pairs),
            'failure_counter_deltas': failure_deltas(candidate),
            'diagnostic_strata': {k: compare(*v) for k, v in strata.items()},
            'diagnostic_warning': 'These strata explain differing work after observing outcomes. They do not replace complete-workflow acceptance or establish causation.'}

decisions = {}
gate = SPEC['selection']
for variant in ['joint', 'offload', 'combined']:
    a = [r for r in rows if r['variant'] == 'baseline']
    b = [r for r in rows if r['variant'] == variant]
    total = compare(a, b)
    checks = {'aggregate_wall': total['wall_speedup_pct'] >= gate['minimum_aggregate_wall_speedup_pct'],
        'all_context_walls': all(reports[str(c)][variant]['overall']['wall_speedup_pct'] >= -gate['maximum_context_wall_regression_pct'] for c in SPEC['contexts']),
        'enough_identical_workflows': all(reports[str(c)][variant]['identical_complete_workflows'] >= gate['minimum_complete_output_identical_workflows_per_context'] for c in SPEC['contexts']),
        'identical_output_wall': all(reports[str(c)][variant]['identical_output_subset'] is not None and reports[str(c)][variant]['identical_output_subset']['wall_speedup_pct'] >= gate['minimum_identical_output_wall_speedup_pct'] for c in SPEC['contexts']),
        'quality': all(reports[str(c)][variant]['additional_behavior_failures'] == 0 for c in SPEC['contexts']),
        'runtime': not failure_deltas(b),
        'memory': total['candidate']['maximum_allocated_bytes'] <= gate['maximum_allocated_bytes'],
        'backup_completion': all(not r['write_completion']['unfinished'] and not r['write_completion']['failed'] for r in b),
        'retention_engaged': not SPEC['variants'][variant]['offload'] or total['candidate']['retentions'] >= gate['minimum_offload_engagement_if_retention_is_deployed']}
    decisions[variant] = {'eligible': all(checks.values()), 'checks': checks, 'aggregate': total}
eligible = [v for v, d in decisions.items() if d['eligible']]
selected = max(eligible, key=lambda v: decisions[v]['aggregate']['wall_speedup_pct']) if eligible else None
if selected in ['combined', 'offload'] and 'joint' in eligible:
    gain = decisions[selected]['aggregate']['wall_speedup_pct'] - decisions['joint']['aggregate']['wall_speedup_pct']
    selected_rows = [r for r in rows if r['variant'] == selected]
    joint_rows = [r for r in rows if r['variant'] == 'joint']
    def p95(items):
        walls = sorted(r['wall_s'] for r in items)
        return walls[math.ceil(.95 * len(walls)) - 1]
    if gain <= .5 and decisions[selected]['aggregate']['candidate']['computed_tokens'] >= decisions['joint']['aggregate']['candidate']['computed_tokens'] and p95(selected_rows) >= p95(joint_rows):
        selected = 'joint'
report = {'scope': SPEC, 'contexts': reports, 'decisions': decisions, 'selected': selected,
    'deployed': False, 'interpretation': 'All256 responses retained. Whole-task observed time can include output changes and shared model failures. No universal speed/accuracy bound. Prior optimization percentages are not added.',
    'files': files}
(OUT / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
(OUT / 'decision.json').write_text(json.dumps({'selected': selected, 'variants': decisions, 'deployed': False}, indent=2) + '\n')
print(json.dumps({'selected': selected, 'variants': decisions}, indent=2))
