"""Shared saved-cohort adapter for the existing advanced model implementations."""
import json
from dataset_paths import recorded_hash

import numpy as np
from sklearn.model_selection import StratifiedKFold

from train_models import ROOT, digest, local_path, read_indexed


def output_paths(cfg, component=None):
    """Keep legacy destinations unless the caller explicitly selects artifactsv2."""
    if cfg.get('input_format') == 'v2':
        root = local_path(cfg['output_root'])
        if root == ROOT / 'artifactsv2' or not root.is_relative_to(ROOT / 'artifactsv2'):
            raise ValueError('output_root must be a child of artifactsv2')
        out = root / cfg['run_name']
        if component:
            out /= component
        return out, out / 'reports'
    out = ROOT / 'artifacts/modeling' / cfg['run_name']
    report = ROOT / 'reports/modeling' / cfg['run_name']
    return (out / component, report / component) if component else (out, report)


def load_v2(cfg):
    folder = local_path(cfg['baseline_run'])
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or cfg['threshold'] != 0.5:
        raise ValueError('A completed v2 baseline and threshold 0.5 are required')
    names = cfg['experiments']
    if not names or len(names) != len(set(names)) or not set(names) <= set(manifest['experiments']):
        raise ValueError('Choose unique experiments recorded in the v2 baseline manifest')
    baseline_cfg = manifest['config']
    schema_path = (local_path(baseline_cfg['prepared_dir']) / 'datasets' /
                   baseline_cfg['policy'] / baseline_cfg['mri_version'] / 'predictor_schema.json')
    if digest(schema_path) != recorded_hash(manifest['input_hashes'], schema_path):
        raise ValueError('Prepared feature schema changed since baseline training')
    allowed = set(json.loads(schema_path.read_text())['predictor_columns'])
    cohorts = {}
    for name in names:
        spec = manifest['experiments'][name]
        path = folder / (spec['cohort'] + '.csv')
        if digest(path) != manifest['artifact_hashes'].get(path.name):
            raise ValueError('Saved cohort, labels, or folds changed: ' + path.name)
        frame = read_indexed(path)
        features = spec['features']
        if (len(features) != len(set(features)) or not set(features) <= allowed
                or set(features) & {'RID', 'PTID', 'label', 'fold', 'baseline_phase'}):
            raise ValueError('Invalid predictor allowlist')
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError('Complete finite predictors are required')
        if not frame.label.isin([0, 1]).all() or set(frame.fold) != set(range(1, 6)):
            raise ValueError('Expected binary labels and five saved participant folds')
        for fold in range(1, 6):
            train, held = frame[frame.fold != fold], frame[frame.fold == fold]
            if (held.label.nunique() != 2 or
                    train.label.value_counts().reindex([0, 1], fill_value=0).min() < cfg['inner_folds']):
                raise ValueError('Insufficient class counts for inner folds')
            if 'stacking_folds' in cfg:
                split = StratifiedKFold(cfg['inner_folds'], shuffle=True, random_state=cfg['seed'] + fold)
                for fitting, _ in split.split(train, train.label):
                    if train.iloc[fitting].label.value_counts().reindex([0, 1], fill_value=0).min() < cfg['stacking_folds']:
                        raise ValueError('Insufficient class counts for nested LR stacking')
        cohorts[name] = frame
    return manifest, cohorts
