#!/usr/bin/env python3
"""Run the remaining V2 experiments sequentially; preserve completed outputs.

Uses only the standard library itself. Training subprocesses use the selected
modeling and pinned TabPFN interpreters. --plan never starts a subprocess.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
METHODS = ('voting', 'stacking', 'lr_stacking', 'mlp', 'resnet', 'hybrid')


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Project input/output paths must stay inside the repository')
    return path


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def interpreter(value):
    if '/' in value:
        path = Path(value).expanduser()
        # Keep the venv executable path: resolving its symlink loses the venv.
        return str(path.absolute())
    return shutil.which(value) or value


def baseline_task(config, phase):
    path = project_path(config)
    cfg = read_json(path)
    return {'name': path.stem, 'phase': phase, 'stage': 'baselines',
            'config': cfg, 'output': project_path(cfg['output_dir']),
            'args': ['src/scripts/train_modelsv2.py', '--config', config]}


def suite_task(suite, method, phase):
    family = ('ensembles' if method in {'voting', 'stacking'} else
              'deep_learning' if method in {'mlp', 'resnet', 'tabpfn'} else
              'hybrid_attention' if method == 'hybrid' else 'lr_stacking')
    folder = 'config/' + suite
    cfg = read_json(project_path(folder) / (family + '.json'))
    output = project_path(cfg['output_root']) / cfg['run_name']
    if method in {'voting', 'stacking', 'mlp', 'resnet', 'tabpfn'}:
        output /= method
    return {'name': suite + '_' + method, 'phase': phase,
            'stage': 'tabpfn' if method == 'tabpfn' else 'advanced',
            'config': cfg, 'output': output,
            'args': ['src/scripts/run_model_suite_v2.py', '--config-dir', folder,
                     '--method', method]}


def build_tasks():
    tasks = [baseline_task('config/modelingv2_biomarkers_primary.json', 'primary')]
    # Include prerequisite baselines. Completed runs are checked and skipped.
    tasks.extend(baseline_task(path, 'primary') for path in
                 ['config/modelingv2.json', 'config/modelingv2_apoe.json'])
    tasks.extend(suite_task(suite, 'tabpfn', 'primary') for suite in
                 ['suite-primary', 'suite-apoe-primary'])
    tasks.extend(baseline_task(path, 'same-day') for path in
                 ['config/modelingv2_biomarkers_same_day.json',
                  'config/modelingv2_same_day.json',
                  'config/modelingv2_apoe_same_day.json'])
    for suite in ['suite-same-day', 'suite-apoe-same-day']:
        tasks.extend(suite_task(suite, method, 'same-day') for method in METHODS)
        tasks.append(suite_task(suite, 'tabpfn', 'same-day'))
    return tasks


def state(task):
    out = task['output']
    if not out.exists() or not any(out.iterdir()):
        return 'pending'
    path = out / 'manifest.json'
    if not path.is_file():
        return 'incomplete'
    manifest = read_json(path)
    if manifest.get('config') != task['config']:
        raise ValueError(f'{out.relative_to(ROOT)}: saved configuration differs; '
                         'restore its config or select a fresh output directory')
    if manifest.get('status') != 'complete':
        return 'incomplete'
    # Historical source hashes may differ after extending the loader. Check the
    # stored result files themselves rather than requiring old code to match.
    for relative in ['held_out_predictions.csv', 'reports/metrics.csv']:
        path = out / relative
        expected = manifest.get('artifact_hashes', {}).get(relative)
        if not expected or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Completed output missing or changed: {path}')
    return 'complete'


def check_dependency(task):
    cfg = task['config']
    if 'baseline_run' not in cfg:
        return
    path = project_path(cfg['baseline_run']) / 'manifest.json'
    if not path.is_file() or read_json(path).get('status') != 'complete':
        raise ValueError(f'{task["name"]}: complete its baseline first: {path}')


def preflight(tasks, interpreters):
    for stage in {task['stage'] for task in tasks}:
        executable = interpreters['tabpfn' if stage == 'tabpfn' else 'standard']
        if not Path(executable).is_file():
            raise ValueError(f'Python interpreter not found: {executable}')
    tab_tasks = [task for task in tasks if task['stage'] == 'tabpfn']
    if tab_tasks:
        # Package metadata only; importing a model or downloading weights is not
        # needed to verify the dedicated environment before a long run.
        versions = subprocess.check_output(
            [interpreters['tabpfn'], '-c',
             'import importlib.metadata as m,json; '
             'print(json.dumps({p:m.version(p) for p in '
             '["tabpfn","scikit-learn","torch"]}))'],
            cwd=ROOT, text=True)
        expected = {'tabpfn': '2.0.9', 'scikit-learn': '1.6.1', 'torch': '2.5.1'}
        if json.loads(versions) != expected:
            raise ValueError('TabPFN environment must match requirements-tabpfn.txt')
        for task in tab_tasks:
            cfg = task['config']['tabpfn']
            checkpoint = project_path(cfg['model_path'])
            if not checkpoint.is_file() or digest(checkpoint) != cfg['checkpoint_sha256']:
                raise ValueError(f'TabPFN checkpoint missing or changed: {checkpoint}')


def write_summary():
    """Collect aggregate metrics with run identity; never pool participants."""
    rows = []
    for path in sorted((ROOT / 'artifactsv2').rglob('reports/metrics.csv')):
        if '_incomplete' in path.parts:
            continue
        manifest_path = path.parent.parent / 'manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest.get('status') != 'complete':
            continue
        cfg = manifest['config']
        specs = manifest.get('experiment_specs') or manifest.get('experiments') or {}
        baseline_cfg = manifest.get('baseline_config', manifest.get('original_config', {}))
        if cfg.get('baseline_run'):
            baseline = read_json(project_path(cfg['baseline_run']) / 'manifest.json')
            specs = specs or baseline['experiments']
            baseline_cfg = baseline['config']
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                # Do not reintroduce archived controls into future result inventories.
                if row.get('model') == 'dummy':
                    continue
                row.update(source=str(path.relative_to(ROOT)),
                           policy=cfg.get('policy', baseline_cfg.get('policy', '')),
                           predictor_count=len(specs.get(row['experiment'], {}).get('features', [])))
                if not row.get('accuracy'):
                    row['accuracy'] = (int(row['tp']) + int(row['tn'])) / int(row['participants'])
                rows.append(row)
    if rows:
        target = ROOT / 'artifactsv2/remaining_models_summary.csv'
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with target.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print('Aggregate inventory:', target.relative_to(ROOT), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable,
                        help='Standard modeling Python (default: current interpreter)')
    parser.add_argument('--tabpfn-python', default=str(ROOT / '.venv-tabpfn-v2/bin/python'))
    parser.add_argument('--phase', choices=['all', 'primary', 'same-day'], default='all')
    parser.add_argument('--stage', choices=['all', 'baselines', 'advanced', 'tabpfn'], default='all')
    parser.add_argument('--plan', action='store_true', help='Print tasks; do not start training or write files')
    parser.add_argument('--retry-incomplete', action='store_true',
                        help='Preserve incomplete outputs in artifactsv2/_incomplete before restarting that stage')
    args = parser.parse_args()
    interpreters = {'standard': interpreter(args.python),
                    'tabpfn': interpreter(args.tabpfn_python)}
    tasks = [task for task in build_tasks()
             if args.phase in {'all', task['phase']} and args.stage in {'all', task['stage']}]
    pending = []
    for task in tasks:
        status = state(task)
        task['state'] = status
        executable = interpreters['tabpfn' if task['stage'] == 'tabpfn' else 'standard']
        task['command'] = [executable, '-u', *task['args']]
        print(f'{status.upper():10} {task["name"]}', flush=True)
        if status != 'complete':
            print('  ' + shlex.join(task['command']), flush=True)
            pending.append(task)
    if args.plan:
        return
    if any(task['state'] == 'incomplete' for task in pending) and not args.retry_incomplete:
        raise ValueError('Incomplete output found. Inspect it, then use --retry-incomplete '
                         'to preserve that output and restart the affected stage.')
    preflight(pending, interpreters)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logs = ROOT / 'artifactsv2/remaining_logs' / stamp
    if pending:
        logs.mkdir(parents=True)
    try:
        for task in pending:
            check_dependency(task)
            if task['state'] == 'incomplete':
                destination = ROOT / 'artifactsv2/_incomplete' / stamp / task['name']
                destination.parent.mkdir(parents=True, exist_ok=True)
                task['output'].rename(destination)
                print('Preserved incomplete run:', destination.relative_to(ROOT), flush=True)
            log_path = logs / (task['name'] + '.log')
            print('\nRUN ' + task['name'] + ' | log: ' + str(log_path.relative_to(ROOT)), flush=True)
            with log_path.open('w') as log:
                process = subprocess.Popen(task['command'], cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if returncode:
                raise RuntimeError(f'{task["name"]} exited with code {returncode}. See {log_path}')
            if state(task) != 'complete':
                raise RuntimeError(f'{task["name"]} did not produce a complete manifest')
    finally:
        write_summary()
    print('\nAll selected stages are complete. Existing completed runs were skipped.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
