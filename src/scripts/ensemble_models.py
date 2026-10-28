"""Shared implementation for the two user-run ensemble notebooks. No automatic execution."""
import itertools
import json
from dataset_paths import recorded_hash
from modeling_v2_inputs import load_v2, output_paths
import platform
import re
import warnings
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from train_models import (ROOT, digest, local_path, make_pipeline, metrics,
                          plt, probability_metrics, read_indexed, save_json)


def load_inputs(cfg):
    baseline = local_path(cfg['baseline_run'])
    manifest = json.loads((baseline / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('The original modeling run must be complete')
    if cfg['threshold'] != 0.5:
        raise ValueError('Keep the prespecified 0.5 threshold for this comparison')
    if set(cfg['base_models']) != {'logistic_regression', 'random_forest', 'xgboost', 'extra_trees'} or len(cfg['base_models']) != 4:
        raise ValueError('These notebooks use the four documented base models')
    if not cfg['experiments'] or len(set(cfg['experiments'])) != len(cfg['experiments']):
        raise ValueError('Choose a nonempty, unique experiment list')
    if cfg['inner_folds'] < 2 or cfg['bootstrap_repetitions'] < 100 or cfg['stacking_C'] <= 0:
        raise ValueError('Invalid inner-fold, bootstrap, or stacking configuration')
    if cfg.get('input_format') == 'v2':
        return load_v2(cfg)
    # Use the feature allowlist recorded by the original run.
    schema_path = local_path(manifest['config']['input_dir']) / 'predictor_schema.json'
    if digest(schema_path) != recorded_hash(manifest['input_hashes'], schema_path):
        raise ValueError('The predictor schema changed since the baseline run')
    allowed = set(json.loads(schema_path.read_text())['predictor_columns'])
    cohorts = {}
    for experiment in cfg['experiments']:
        if experiment not in {'clinical_reference', 'matched_clinical', 'matched_fusion'}:
            raise ValueError('Only the three documented clinical/fusion experiments are supported')
        spec = manifest['experiments'][experiment]
        features = spec['features']
        if len(features) != len(set(features)) or not set(features) <= allowed:
            raise ValueError('Features differ from the predictor allowlist')
        filename = spec['cohort'] + '.csv'
        if digest(baseline / filename) != manifest['artifact_hashes'][filename]:
            raise ValueError(f'Baseline cohort changed: {filename}')
        frame = read_indexed(baseline / filename)
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError('Selected predictors must be finite and complete')
        if not frame.label.isin([0, 1]).all() or set(frame.fold) != set(range(1, 6)):
            raise ValueError('Expected binary labels and the five saved outer folds')
        for fold in range(1, 6):
            counts = frame.loc[frame.fold != fold, 'label'].value_counts().reindex([0, 1], fill_value=0)
            if counts.min() < cfg['inner_folds']:
                raise ValueError('Insufficient training class counts for inner cross-validation')
        cohorts[experiment] = frame
    return manifest, cohorts


def base_pipeline(name, features, baseline_cfg, cfg):
    if name != 'extra_trees':
        return make_pipeline(name, features, baseline_cfg)
    pipeline = make_pipeline('logistic_regression', features, baseline_cfg)
    pipeline.set_params(model=ExtraTreesClassifier(random_state=cfg['seed'], **cfg['extra_trees']))
    return pipeline


def choose_weights(probabilities, labels, step):
    units = round(1 / step) if step > 0 else 0
    if units < 1 or units > 10 or not np.isclose(units * step, 1):
        raise ValueError('Weight step must evenly divide 1, with at most ten units')
    count = probabilities.shape[1]
    # Include equal weights and pure single-model choices as well as the small grid.
    candidates = [np.ones(count) / count]
    candidates += [np.asarray(w, dtype=float) / units
                   for w in itertools.product(range(units + 1), repeat=count) if sum(w) == units]
    losses = [log_loss(labels, probabilities @ weights, labels=[0, 1]) for weights in candidates]
    best = int(np.argmin(losses))
    return candidates[best], float(losses[best])


def scoring(y, probability):
    return {**probability_metrics(y, probability),
            'accuracy': float(np.mean((np.asarray(probability) >= 0.5) == np.asarray(y)))}


def paired_intervals(y, candidate, reference, cfg):
    y, candidate, reference = map(np.asarray, (y, candidate, reference))
    point, base = scoring(y, candidate), scoring(y, reference)
    differences = {key: [] for key in point}
    rng = np.random.default_rng(cfg['seed'])
    for _ in range(cfg['bootstrap_repetitions']):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        a, b = scoring(y[indices], candidate[indices]), scoring(y[indices], reference[indices])
        for key in differences:
            differences[key].append(a[key] - b[key])
    return [{'metric': key, 'difference': point[key] - base[key],
             'lower_95': float(np.quantile(values, 0.025)),
             'upper_95': float(np.quantile(values, 0.975)), 'bootstrap_samples': len(values)}
            for key, values in differences.items()]


def report_results(predictions, cfg, report, ensemble_names):
    rows, folds, paired = [], [], []
    for (experiment, model), group in predictions.groupby(['experiment', 'model'], sort=False):
        key = {'experiment': experiment, 'model': model}
        rows.append({**key, **metrics(group.label, group.probability, cfg['threshold']),
                     'accuracy': float(np.mean(group.label == group.prediction))})
        for fold, part in group.groupby('fold'):
            folds.append({**key, 'fold': int(fold), **metrics(part.label, part.probability, cfg['threshold']),
                          'accuracy': float(np.mean(part.label == part.prediction))})
        if model in ensemble_names:
            reference = predictions[(predictions.experiment == experiment) &
                                    (predictions.model == 'logistic_regression')]
            joined = group.merge(reference, on='RID', suffixes=('_ensemble', '_reference'), validate='one_to_one')
            if (len(joined) != len(group) or len(joined) != len(reference) or
                    not joined.label_ensemble.eq(joined.label_reference).all() or
                    not joined.fold_ensemble.eq(joined.fold_reference).all()):
                raise ValueError('Paired comparisons require identical participants, labels, and folds')
            paired.extend({**key, 'reference': 'logistic_regression', **row} for row in
                          paired_intervals(joined.label_ensemble, joined.probability_ensemble,
                                           joined.probability_reference, cfg))
    results = pd.DataFrame(rows)
    results.to_csv(report / 'metrics.csv', index=False)
    pd.DataFrame(folds).to_csv(report / 'fold_metrics.csv', index=False)
    pd.DataFrame(paired).to_csv(report / 'paired_vs_logistic.csv', index=False)
    for experiment, groups in predictions.groupby('experiment', sort=False):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Ideal')
        for name, group in groups.groupby('model', sort=False):
            observed, probability = calibration_curve(group.label, group.probability, n_bins=5, strategy='quantile')
            ax.plot(probability, observed, 'o-', label=name)
        ax.set(xlabel='Mean held-out probability', ylabel='Observed AD fraction', title=experiment,
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(report / f'calibration_{experiment}.png', dpi=160)
        plt.close(fig)
    lines = ['# Exploratory ensemble results', '',
             '| Experiment | Model | N | Accuracy | AUROC | Average precision | Brier |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['model']} | {row['participants']} | "
                     f"{row['accuracy']:.1%} | {row['auroc']:.3f} | {row['average_precision']:.3f} | {row['brier_score']:.3f} |")
    lines += ['', '## Reading these results', '',
              '- Accuracy is the fraction of correct held-out predictions at threshold 0.5. All CSV rates use the 0-1 scale.',
              '- Average precision is the non-interpolated AUPRC summary. Higher AUROC/AP and lower Brier are better.',
              '- The four base models are refitted in this environment; compare ensembles with these same-run baselines.',
              '- Saved original outer folds are reused. Weights or stacking coefficients are learned only within outer training participants.',
              '- The earlier results motivated these experiments. They are exploratory reuse of an already inspected cohort, not independent confirmation.',
              '- Paired differences are ensemble minus same-run Logistic Regression. Intervals condition on saved predictions, omit retraining uncertainty, and are not adjusted for multiple comparisons.',
              '- Do not keep changing methods against these results and call the best score an unbiased final estimate.',
              '- Compare methods on identical saved participants and folds; cohort composition is recorded in the baseline manifest. Complete-case selection and provisional outcomes limit generalization.',
              '- No higher score is guaranteed. An ensemble that does not beat Logistic Regression is a valid result.', '']
    (report / 'results.md').write_text('\n'.join(lines))
    return results


def run_experiment(cfg, method):
    if method not in {'voting', 'stacking'}:
        raise ValueError('Choose voting or stacking')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_name']):
        raise ValueError('Use only letters, digits, underscores, and hyphens in run_name')
    original, cohorts = load_inputs(cfg)
    baseline = local_path(cfg['baseline_run'])
    out, report = output_paths(cfg, method)
    for folder in [out, report]:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('This experiment already has outputs. Change run_name for a new run, or display the existing results without rerunning training.')
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    manifest = {'status': 'running', 'method': method, 'config': cfg,
                'baseline_config': original['config'], 'python': platform.python_version(),
                'baseline_manifest_sha256': digest(baseline / 'manifest.json'),
                'script_sha256': digest(ROOT / 'src/scripts' / 'ensemble_models.py'),
                'shared_training_script_sha256': digest(ROOT / 'src/scripts' / 'train_models.py'),
                'versions': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib', 'matplotlib']},
                'cohorts': original['cohorts'],
                'experiments': {name: original['experiments'][name] for name in cfg['experiments']},
                'evaluation': 'Exploratory reuse of original outer folds; inner out-of-fold training for ensembles.'}
    manifest['input_adapter_sha256'] = digest(ROOT / 'src/scripts/modeling_v2_inputs.py')
    save_json(out / 'manifest.json', manifest)
    ensemble_names = ['equal_voting', 'weighted_voting'] if method == 'voting' else ['stacking']
    predictions, selections = [], []
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            for experiment, frame in cohorts.items():
                features = original['experiments'][experiment]['features']
                frame[features + ['label', 'fold']].to_csv(out / f'{experiment}_cohort.csv', index_label='RID')
                for fold in range(1, 6):
                    print(f'{method}: {experiment}, outer fold {fold}/5', flush=True)
                    train, held = frame[frame.fold != fold], frame[frame.fold == fold]
                    inner = list(StratifiedKFold(n_splits=cfg['inner_folds'], shuffle=True,
                                                random_state=cfg['seed'] + fold).split(train, train.label))
                    inner_probabilities, held_probabilities, fitted = [], [], []
                    inner_folds = np.zeros(len(train), dtype=int)
                    for inner_fold, (_, val) in enumerate(inner, 1):
                        inner_folds[val] = inner_fold
                    for name in cfg['base_models']:
                        pipeline = base_pipeline(name, features, original['config'], cfg)
                        # Each base estimator includes preprocessing, fitted separately in each inner fold.
                        inner_probabilities.append(cross_val_predict(pipeline, train[features], train.label,
                                                                     cv=inner, method='predict_proba', n_jobs=1)[:, 1])
                        pipeline.fit(train[features], train.label)
                        held_probabilities.append(pipeline.predict_proba(held[features])[:, 1])
                        fitted.append(pipeline)
                    inner_matrix, held_matrix = np.column_stack(inner_probabilities), np.column_stack(held_probabilities)
                    audit = pd.DataFrame(inner_matrix, index=train.index, columns=cfg['base_models'])
                    audit['label'], audit['inner_fold'] = train.label, inner_folds
                    audit.to_csv(out / f'{experiment}_fold{fold}_inner_predictions.csv', index_label='RID')
                    probabilities = dict(zip(cfg['base_models'], held_matrix.T))
                    bundle = {'base_names': cfg['base_models'], 'base_pipelines': fitted, 'features': features,
                              'method': method, 'threshold': cfg['threshold']}
                    if method == 'voting':
                        weights, loss = choose_weights(inner_matrix, train.label, cfg['voting_weight_step'])
                        probabilities['equal_voting'] = held_matrix.mean(axis=1)
                        probabilities['weighted_voting'] = held_matrix @ weights
                        bundle['weights'] = weights
                        selections.append({'experiment': experiment, 'fold': fold, 'inner_log_loss': loss,
                                           **dict(zip(cfg['base_models'], weights.tolist()))})
                    else:
                        # The meta-model sees cross-fitted probabilities, never in-sample base predictions.
                        meta = LogisticRegression(C=cfg['stacking_C'], max_iter=3000, random_state=cfg['seed'])
                        meta.fit(inner_matrix, train.label)
                        probabilities['stacking'] = meta.predict_proba(held_matrix)[:, 1]
                        bundle['meta_model'] = meta
                        selections.append({'experiment': experiment, 'fold': fold, 'intercept': float(meta.intercept_[0]),
                                           **dict(zip(cfg['base_models'], meta.coef_[0].tolist()))})
                    joblib.dump(bundle, out / 'models' / f'{experiment}_fold{fold}.joblib')
                    for name, probability in probabilities.items():
                        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
                            raise ValueError('Invalid ensemble probabilities')
                        predictions.extend({'experiment': experiment, 'model': name, 'fold': fold, 'RID': rid,
                                            'label': int(label), 'probability': float(p), 'prediction': int(p >= cfg['threshold'])}
                                           for rid, label, p in zip(held.index, held.label, probability))
        prediction_frame = pd.DataFrame(predictions)
        prediction_frame.to_csv(out / 'held_out_predictions.csv', index=False)
        pd.DataFrame(selections).to_csv(report / 'ensemble_parameters.csv', index=False)
        print('Writing metrics and paired bootstrap comparisons.', flush=True)
        results = report_results(prediction_frame, cfg, report, ensemble_names)
        manifest['status'] = 'complete'
        manifest['artifact_hashes'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                                      if p.is_file() and p.name != 'manifest.json'}
    except Exception as exc:
        manifest['status'], manifest['error'] = 'failed', f'{type(exc).__name__}: {exc}'
        save_json(out / 'manifest.json', manifest)
        raise
    save_json(out / 'manifest.json', manifest)
    print(f'Finished: {report.relative_to(ROOT)}', flush=True)
    return results
