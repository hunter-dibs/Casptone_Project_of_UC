#!/usr/bin/env python3
"""Train the prespecified complete-case experiments. Run manually, locally or in Colab."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from dataset_paths import project_path
import platform
import warnings

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEX = 'PTDEMOG__PTGENDER'
BOOT_METRICS = ('auroc', 'average_precision', 'brier_score')


def local_path(value):
    return project_path(value)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_indexed(path):
    frame = pd.read_csv(path, dtype={'RID': str, 'PTID': str})
    if frame['RID'].isna().any() or frame['RID'].duplicated().any():
        raise ValueError(f'{path.name}: RID must be present and unique')
    return frame.set_index('RID').sort_index()


def assemble(cfg):
    folder = local_path(cfg['input_dir'])
    schema = json.loads((folder / 'predictor_schema.json').read_text())
    predictors = read_indexed(folder / 'baseline_predictors.csv')
    outcomes = read_indexed(folder / 'outcome_audit.csv')
    if set(predictors.index) != set(outcomes.index):
        raise ValueError('Predictor and outcome participant sets differ')
    outcomes = outcomes.reindex(predictors.index)
    if not predictors['PTID'].equals(outcomes['PTID']):
        raise ValueError('Participant identifiers disagree between input files')
    clinical = schema['modalities']['clinical']
    apoe = schema['modalities']['APOE']
    features = clinical + apoe
    if len(set(features)) != len(features) or not set(features) <= set(schema['predictor_columns']):
        raise ValueError('Clinical/APOE features must be unique allowlisted predictors')
    forbidden = set(outcomes.columns) | {'RID', 'PTID'}
    if set(features) & forbidden:
        raise ValueError('Identifier or outcome information appears in predictor columns')
    x = predictors[features].apply(pd.to_numeric, errors='raise')
    if np.isinf(x.to_numpy()).any():
        raise ValueError('Predictors contain infinite values')
    y = pd.to_numeric(outcomes['label_provisional'], errors='raise')
    if not y.dropna().isin([0, 1]).all():
        raise ValueError('Unexpected outcome label')
    prepared = json.loads(local_path(cfg['preparation_manifest']).read_text())
    preparation_cfg = prepared['config']
    if (preparation_cfg['horizon_months'] != 36 or
            preparation_cfg['availability_policy'] != 'result_by_baseline' or
            preparation_cfg['apoe_policy'] != 'tested_by_baseline' or
            preparation_cfg['followup_tolerance_days'] != 0 or
            any(days != 90 for days in preparation_cfg['lookback_days'].values())):
        raise ValueError('This experiment expects the documented strict 36-month preparation')
    if not outcomes['horizon_months'].eq(36).all():
        raise ValueError('Outcome horizon differs from the prespecified experiment')
    eligible = y.notna()
    cohort_masks = {
        'clinical_complete': eligible & x[clinical].notna().all(axis=1),
        'matched_complete': eligible & x[features].notna().all(axis=1),
    }
    cohorts, summaries = {}, []
    for name, mask in cohort_masks.items():
        frame = x.loc[mask].copy()
        frame['label'] = y.loc[mask].astype(int)
        frame['baseline_phase'] = outcomes.loc[mask, 'baseline_phase']
        if frame['label'].value_counts().reindex([0, 1], fill_value=0).min() < cfg['folds']:
            raise ValueError(f'{name}: insufficient class counts for the requested folds')
        fold = pd.Series(0, index=frame.index, dtype=int)
        splitter = StratifiedKFold(n_splits=cfg['folds'], shuffle=True, random_state=cfg['seed'])
        for number, (_, held) in enumerate(splitter.split(frame, frame['label']), 1):
            fold.iloc[held] = number
        frame['fold'] = fold
        cohorts[name] = frame
        summaries.append({'cohort': name, 'participants': len(frame),
                          'converters': int(frame['label'].sum()),
                          'nonconverters': int((frame['label'] == 0).sum()),
                          'phases': frame['baseline_phase'].value_counts().to_dict()})
    experiments = {
        'clinical_reference': ('clinical_complete', clinical),
        'matched_clinical': ('matched_complete', clinical),
        'matched_apoe': ('matched_complete', apoe),
        'matched_fusion': ('matched_complete', features),
    }
    return cohorts, summaries, experiments, outcomes


def make_pipeline(name, features, cfg):
    numeric = [feature for feature in features if feature != SEX]
    transforms = [('numeric', StandardScaler(), numeric)]
    if SEX in features:
        transforms.append(('sex', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [SEX]))
    preprocessing = ColumnTransformer(transforms, remainder='drop', verbose_feature_names_out=False)
    params = cfg['parameters'].get(name, {})
    if name == 'logistic_regression':
        model = LogisticRegression(random_state=cfg['seed'], **params)
    elif name == 'random_forest':
        model = RandomForestClassifier(random_state=cfg['seed'], **params)
    elif name == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              random_state=cfg['seed'], **params)
    else:
        raise ValueError(f'Unknown model: {name}')
    return Pipeline([('preprocessing', preprocessing), ('model', model)])


def probability_metrics(y, probability):
    return {'auroc': float(roc_auc_score(y, probability)),
            'average_precision': float(average_precision_score(y, probability)),
            'brier_score': float(brier_score_loss(y, probability))}


def metrics(y, probability, threshold):
    prediction = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {**probability_metrics(y, probability),
            'sensitivity': float(recall_score(y, prediction, zero_division=0)),
            'specificity': float(tn / (tn + fp)),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'participants': len(y), 'prevalence': float(np.mean(y)), 'threshold': threshold}


def bootstrap(y, probability, cfg, reference=None):
    """Resample participants; paired comparisons use identical sampled indices."""
    y, probability = np.asarray(y), np.asarray(probability)
    reference = None if reference is None else np.asarray(reference)
    rng = np.random.default_rng(cfg['seed'])
    samples = {key: [] for key in BOOT_METRICS}
    point = probability_metrics(y, probability)
    if reference is not None:
        base = probability_metrics(y, reference)
        point = {key: point[key] - base[key] for key in point}
    for _ in range(cfg['bootstrap_repetitions']):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        result = probability_metrics(y[idx], probability[idx])
        if reference is not None:
            base = probability_metrics(y[idx], reference[idx])
            result = {key: result[key] - base[key] for key in result}
        for key in samples:
            samples[key].append(result[key])
    rows = []
    for key, values in samples.items():
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({'metric': key, 'estimate': point[key], 'lower_95': float(low),
                     'upper_95': float(high), 'bootstrap_samples': len(values)})
    return rows


def write_reports(predictions, cfg, report):
    aggregate, fold_results, intervals, paired, calibration = [], [], [], [], []
    for (experiment, model), group in predictions.groupby(['experiment', 'model'], sort=False):
        keys = {'experiment': experiment, 'model': model}
        aggregate.append({**keys, **metrics(group.label, group.probability, cfg['threshold'])})
        intervals.extend({**keys, **row} for row in bootstrap(group.label, group.probability, cfg))
        for fold, part in group.groupby('fold'):
            fold_results.append({**keys, 'fold': fold,
                                 **metrics(part.label, part.probability, cfg['threshold'])})
        observed, predicted = calibration_curve(group.label, group.probability, n_bins=5, strategy='quantile')
        calibration.extend({**keys, 'mean_probability': float(p), 'observed_fraction': float(o)}
                           for p, o in zip(predicted, observed))
    for model in cfg['models']:
        a = predictions[(predictions.experiment == 'matched_clinical') & (predictions.model == model)]
        b = predictions[(predictions.experiment == 'matched_fusion') & (predictions.model == model)]
        joined = a.merge(b, on='RID', suffixes=('_clinical', '_fusion'), validate='one_to_one')
        if len(joined) != len(a) or len(joined) != len(b):
            raise ValueError('Paired comparison membership differs')
        if not (joined.label_clinical.eq(joined.label_fusion).all() and
                joined.fold_clinical.eq(joined.fold_fusion).all()):
            raise ValueError('Paired comparison labels or folds differ')
        paired.extend({'model': model, 'comparison': 'fusion_minus_clinical', **row}
                      for row in bootstrap(joined.label_fusion, joined.probability_fusion, cfg,
                                           reference=joined.probability_clinical))
    results = pd.DataFrame(aggregate)
    for name, rows in [('metrics', aggregate), ('fold_metrics', fold_results),
                       ('confidence_intervals', intervals), ('paired_differences', paired),
                       ('calibration', calibration)]:
        pd.DataFrame(rows).to_csv(report / f'{name}.csv', index=False)
    pd.DataFrame(fold_results).groupby(['experiment', 'model'])[list(BOOT_METRICS)].agg(
        ['mean', 'std']).to_csv(report / 'fold_variation.csv')
    for experiment, groups in predictions.groupby('experiment', sort=False):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Ideal')
        for model, group in groups.groupby('model', sort=False):
            observed, predicted = calibration_curve(group.label, group.probability, n_bins=5, strategy='quantile')
            ax.plot(predicted, observed, 'o-', label=model)
        ax.set(xlabel='Mean held-out probability', ylabel='Observed AD fraction',
               title=experiment.replace('_', ' '), xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(report / f'calibration_{experiment}.png', dpi=160)
        plt.close(fig)
    lines = ['# Provisional complete-case modeling results', '',
             'These are held-out cross-validation results using the provisional 36-month endpoint.', '',
             '| Experiment | Model | N | AUROC | AP | Brier |',
             '|---|---|---:|---:|---:|---:|']
    for row in results.to_dict('records'):
        lines.append(f"| {row['experiment']} | {row['model']} | {row['participants']} | "
                     f"{row['auroc']:.3f} | {row['average_precision']:.3f} | {row['brier_score']:.3f} |")
    lines += ['', '## Interpretation and limitations', '',
              '- AP is non-interpolated average precision, used here as the AUPRC summary; it is not trapezoidal PR area.',
              '- Metrics are proportions on a 0-1 scale; confusion-matrix entries are counts. Undefined precision/F1 are recorded as zero.',
              '- Five-fold evaluation uses fixed settings without tuning. Preprocessing is fitted within each training fold.',
              '- The broader clinical cohort and matched cohort are different populations. Compare modalities only within the matched cohort.',
              '- See cohort_summary.json for actual phase composition. The original matched cohort is entirely ADNI1.',
              '- Complete-case selection and exclusion of unresolved outcomes can introduce selection bias.',
              '- Bootstrap intervals condition on the saved out-of-fold predictions; they do not capture all retraining uncertainty.',
              '- Paired differences are fusion minus clinical: positive AUROC/AP favors fusion; negative Brier favors fusion.',
              '- Do not treat the best observed model as independently validated after selecting it from this comparison.',
              '- MRI/CSF comparisons remain unsupported. These results do not answer the full four-modality research question.',
              '- SHAP is generated separately; no clinical effectiveness or causal conclusions are established.', '']
    (report / 'results.md').write_text('\n'.join(lines))


def main(config='config/modeling.json'):
    config_path = local_path(config)
    cfg = json.loads(config_path.read_text())
    if cfg['folds'] != 5 or cfg['threshold'] != 0.5 or cfg['bootstrap_repetitions'] < 100:
        raise ValueError('This protocol requires five folds, threshold 0.5, and at least 100 bootstrap draws')
    known = {'logistic_regression', 'random_forest', 'xgboost'}
    if not cfg['models'] or not set(cfg['models']) <= known or len(set(cfg['models'])) != len(cfg['models']):
        raise ValueError('Specify a nonempty unique list of supported models')
    out, report = local_path(cfg['output_dir']), local_path(cfg['report_dir'])
    if not out.is_relative_to(ROOT / 'artifacts' / 'modeling') or not report.is_relative_to(ROOT / 'reports' / 'modeling'):
        raise ValueError('Outputs must stay in artifacts/modeling and reports/modeling')
    for folder in [out, report]:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f'{folder.relative_to(ROOT)} already contains a run; choose new output/report directories in a copied config')
    if 'xgboost' in cfg['models']:
        from xgboost import XGBClassifier  # Fail before starting if the dependency is unavailable.
    cohorts, summaries, experiments, outcomes = assemble(cfg)
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    inputs = [local_path(cfg['input_dir']) / name for name in
              ['baseline_predictors.csv', 'outcome_audit.csv', 'predictor_schema.json']]
    inputs += [local_path(cfg['preparation_manifest']), config_path, Path(__file__).resolve()]
    manifest = {'status': 'running', 'config': cfg, 'python': platform.python_version(),
                'input_hashes': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
                'versions': {name: importlib.metadata.version(name) for name in
                             ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'joblib']},
                'experiments': {name: {'cohort': cohort, 'features': features}
                                for name, (cohort, features) in experiments.items()},
                'cohorts': summaries}
    if 'xgboost' in cfg['models']:
        manifest['versions']['xgboost'] = importlib.metadata.version('xgboost')
    save_json(out / 'manifest.json', manifest)
    save_json(report / 'cohort_summary.json', summaries)
    for name, cohort in cohorts.items():
        cohort.to_csv(out / f'{name}.csv', index_label='RID')
    exclusions = []
    for name, frame in cohorts.items():
        for rid in outcomes.index:
            if rid not in frame.index:
                reason = 'unresolved_outcome' if pd.isna(outcomes.loc[rid, 'label_provisional']) else 'incomplete_required_predictors'
                exclusions.append({'cohort': name, 'RID': rid, 'reason': reason})
    pd.DataFrame(exclusions).to_csv(out / 'cohort_exclusions.csv', index=False)
    all_predictions = []
    try:
        # Treat non-convergence as a failed experiment rather than reporting unreliable coefficients.
        warnings.filterwarnings('error', category=ConvergenceWarning)
        for experiment, (cohort_name, features) in experiments.items():
            frame = cohorts[cohort_name]
            for model_name in cfg['models']:
                print(f'Training {experiment}: {model_name}, N={len(frame)}', flush=True)
                for fold in range(1, cfg['folds'] + 1):
                    train, held = frame[frame.fold != fold], frame[frame.fold == fold]
                    pipeline = make_pipeline(model_name, features, cfg)
                    pipeline.fit(train[features], train.label)
                    probability = pipeline.predict_proba(held[features])[:, 1]
                    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
                        raise ValueError('Model returned invalid probabilities')
                    joblib.dump(pipeline, out / 'models' / f'{experiment}__{model_name}__fold{fold}.joblib')
                    for rid, label, p in zip(held.index, held.label, probability):
                        all_predictions.append({'RID': rid, 'experiment': experiment, 'model': model_name,
                                                'fold': fold, 'label': int(label), 'probability': float(p),
                                                'prediction': int(p >= cfg['threshold'])})
        predictions = pd.DataFrame(all_predictions)
        predictions.to_csv(out / 'held_out_predictions.csv', index=False)
        print('Writing metrics, bootstrap intervals, paired comparisons, and calibration figures.', flush=True)
        write_reports(predictions, cfg, report)
        manifest['status'] = 'complete'
        manifest['artifact_hashes'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                                      if p.is_file() and p.name != 'manifest.json'}
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        save_json(out / 'manifest.json', manifest)
        raise
    save_json(out / 'manifest.json', manifest)
    print(f'Finished. Aggregate report: {report.relative_to(ROOT) / "results.md"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/modeling.json')
    main(parser.parse_args().config)
