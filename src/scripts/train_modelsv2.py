#!/usr/bin/env python3
"""Train matched modality comparisons from completed multimodal preparation outputs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold

from train_models import (ROOT, digest, local_path, make_pipeline, metrics,
                          plt, probability_metrics, read_indexed, save_json)


COHORTS = {
    'mri_pair': ['clinical', 'MRI'],
    'csf_pair': ['clinical', 'CSF'],
    'trimodal': ['clinical', 'MRI', 'CSF'],
}
APOE_COHORTS = {
    'apoe_pair': ['clinical', 'APOE'],
    'apoe_csf': ['clinical', 'APOE', 'CSF'],
}
BIOMARKER_COHORTS = {
    'mri_available': ['MRI'],
    'csf_available': ['CSF'],
    'mri_csf_available': ['MRI', 'CSF'],
    'apoe_csf_available': ['APOE', 'CSF'],
}
COHORT_SETS = {'mri_csf': COHORTS, 'apoe': APOE_COHORTS,
               'biomarkers': BIOMARKER_COHORTS}
EXPERIMENTS = {
    'mri_clinical': ('mri_pair', ['clinical']),
    'mri_fusion': ('mri_pair', ['clinical', 'MRI']),
    'csf_clinical': ('csf_pair', ['clinical']),
    'csf_fusion': ('csf_pair', ['clinical', 'CSF']),
    'trimodal_clinical': ('trimodal', ['clinical']),
    'trimodal_clinical_mri': ('trimodal', ['clinical', 'MRI']),
    'trimodal_clinical_csf': ('trimodal', ['clinical', 'CSF']),
    'trimodal_fusion': ('trimodal', ['clinical', 'MRI', 'CSF']),
    'apoe_clinical': ('apoe_pair', ['clinical']),
    'apoe_fusion': ('apoe_pair', ['clinical', 'APOE']),
    'apoe_csf_clinical': ('apoe_csf', ['clinical']),
    'apoe_csf_clinical_apoe': ('apoe_csf', ['clinical', 'APOE']),
    'apoe_csf_clinical_csf': ('apoe_csf', ['clinical', 'CSF']),
    'apoe_csf_fusion': ('apoe_csf', ['clinical', 'APOE', 'CSF']),
    'biomarker_mri': ('mri_available', ['MRI']),
    'biomarker_csf': ('csf_available', ['CSF']),
    'biomarker_mri_csf_mri': ('mri_csf_available', ['MRI']),
    'biomarker_mri_csf_csf': ('mri_csf_available', ['CSF']),
    'biomarker_mri_csf_fusion': ('mri_csf_available', ['MRI', 'CSF']),
    'biomarker_apoe_csf_apoe': ('apoe_csf_available', ['APOE']),
    'biomarker_apoe_csf_csf': ('apoe_csf_available', ['CSF']),
    'biomarker_apoe_csf_fusion': ('apoe_csf_available', ['APOE', 'CSF']),
}
COMPARISONS = [
    ('biomarker_mri_csf_fusion', 'biomarker_mri_csf_mri'),
    ('biomarker_mri_csf_fusion', 'biomarker_mri_csf_csf'),
    ('biomarker_apoe_csf_fusion', 'biomarker_apoe_csf_apoe'),
    ('biomarker_apoe_csf_fusion', 'biomarker_apoe_csf_csf'),
    ('apoe_fusion', 'apoe_clinical'),
    ('apoe_csf_clinical_apoe', 'apoe_csf_clinical'),
    ('apoe_csf_clinical_csf', 'apoe_csf_clinical'),
    ('apoe_csf_fusion', 'apoe_csf_clinical'),
    ('apoe_csf_fusion', 'apoe_csf_clinical_apoe'),
    ('apoe_csf_fusion', 'apoe_csf_clinical_csf'),
    ('mri_fusion', 'mri_clinical'), ('csf_fusion', 'csf_clinical'),
    ('trimodal_clinical_mri', 'trimodal_clinical'),
    ('trimodal_clinical_csf', 'trimodal_clinical'),
    ('trimodal_fusion', 'trimodal_clinical'),
    ('trimodal_fusion', 'trimodal_clinical_mri'),
    ('trimodal_fusion', 'trimodal_clinical_csf'),
]
BOOT_METRICS = ['accuracy', 'auroc', 'average_precision', 'brier_score']


def assemble(cfg):
    prepared = local_path(cfg['prepared_dir'])
    preparation = json.loads((prepared / 'manifest.json').read_text())
    if preparation.get('status') != 'complete':
        raise ValueError('Run multimodal preparation successfully first')
    policy = cfg['policy']
    if policy not in {'retrospective', 'retrospective_same_day'} or cfg['mri_version'] != 'fs51':
        raise ValueError('This experiment uses separate retrospective FS5.1 scenarios')
    expected = {'last_collection_day': -1 if policy == 'retrospective' else 0,
                'processing_by_baseline': False}
    if preparation['policies'][policy] != expected or preparation['config']['lookback_days'] != 90:
        raise ValueError('Preparation policy differs from this protocol')
    dataset = prepared / 'datasets' / policy / cfg['mri_version']
    inputs = [prepared / 'manifest.json']

    def verified(path):
        key = str(path.relative_to(prepared))
        if preparation['output_sha256'].get(key) != digest(path):
            raise ValueError('Prepared file changed or is not registered: ' + key)
        inputs.append(path)
        return path

    schema = json.loads(verified(dataset / 'predictor_schema.json').read_text())
    if schema['scenario'] != policy + '/fs51' or schema['mri_product'] != 'UCSFFSX51':
        raise ValueError('Unexpected scenario schema')
    outcomes = read_indexed(verified(prepared / 'outcome_audit.csv'))
    master = read_indexed(verified(dataset / 'baseline_predictors.csv'))
    if set(master.index) != set(outcomes.index):
        raise ValueError('Prepared master and outcomes have different participants')
    outcomes = outcomes.reindex(master.index)
    if not master.PTID.equals(outcomes.PTID) or not outcomes.horizon_months.eq(36).all():
        raise ValueError('Participant identities or outcome horizon disagree')
    modalities = schema['modalities']
    if [len(modalities[m]) for m in ['clinical', 'APOE', 'MRI', 'CSF']] != [17, 1, 7, 3]:
        raise ValueError('Expected 17 clinical, 1 APOE, 7 MRI and 3 core CSF predictors')
    cohort_set = cfg.get('cohort_set', 'mri_csf')
    if cohort_set not in COHORT_SETS:
        raise ValueError('cohort_set must be mri_csf, apoe or biomarkers')
    selected_cohorts = COHORT_SETS[cohort_set]
    cohorts, summaries = {}, []
    for name, bundle in selected_cohorts.items():
        features = [feature for modality in bundle for feature in modalities[modality]]
        if (len(set(features)) != len(features) or not set(features) <= set(schema['predictor_columns'])
                or set(features) & (set(outcomes.columns) | {'RID', 'PTID', 'label', 'fold'})):
            raise ValueError('Invalid predictor allowlist')
        frame = read_indexed(verified(dataset / 'complete_cases' / ('_'.join(bundle) + '.csv')))
        if set(frame.columns) != {'PTID', 'label', *features}:
            raise ValueError('Complete-case export has unexpected columns')
        y = pd.to_numeric(outcomes.label_provisional, errors='raise')
        expected_ids = master.index[y.notna() & master[features].notna().all(axis=1)]
        if set(frame.index) != set(expected_ids):
            raise ValueError('Complete-case export membership disagrees with the master')
        if not frame.PTID.equals(master.loc[frame.index, 'PTID']):
            raise ValueError('Complete-case participant identity mismatch')
        frame[features] = frame[features].apply(pd.to_numeric, errors='raise')
        original = master.loc[frame.index, features].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(frame[features].to_numpy()).all() or not np.array_equal(frame[features].to_numpy(), original.to_numpy()):
            raise ValueError('Export predictors are incomplete or differ from master values')
        frame['label'] = pd.to_numeric(frame.label, errors='raise')
        if not frame.label.isin([0, 1]).all() or not np.array_equal(frame.label.to_numpy(), y.loc[frame.index].to_numpy()):
            raise ValueError('Export labels differ from frozen outcomes')
        frame['label'] = frame.label.astype(int)
        if frame.label.value_counts().reindex([0, 1], fill_value=0).min() < cfg['folds']:
            raise ValueError(f'{name}: insufficient class counts for {cfg["folds"]} folds')
        frame['baseline_phase'] = outcomes.loc[frame.index, 'baseline_phase']
        frame['fold'] = 0
        splitter = StratifiedKFold(n_splits=cfg['folds'], shuffle=True, random_state=cfg['seed'])
        for fold, (_, held) in enumerate(splitter.split(frame, frame.label), 1):
            frame.loc[frame.index[held], 'fold'] = fold
        cohorts[name] = frame
        summaries.append({'cohort': name, 'modalities': bundle, 'participants': len(frame),
                          'converters': int(frame.label.sum()), 'nonconverters': int((frame.label == 0).sum()),
                          'phases': {str(k): int(v) for k, v in frame.baseline_phase.value_counts().items()}})
    experiments = {name: {'cohort': cohort, 'features': [feature for modality in bundle for feature in modalities[modality]]}
                   for name, (cohort, bundle) in EXPERIMENTS.items() if cohort in selected_cohorts}
    return cohorts, summaries, experiments, inputs


def all_metrics(y, probability, threshold):
    values = metrics(y, probability, threshold)
    values['accuracy'] = (values['tp'] + values['tn']) / values['participants']
    return values


def bootstrap(y, probability, cfg, reference=None):
    y, probability = np.asarray(y), np.asarray(probability)
    reference = None if reference is None else np.asarray(reference)
    def score(labels, values):
        return {**probability_metrics(labels, values),
                'accuracy': float(np.mean((values >= cfg['threshold']) == labels))}
    point = score(y, probability)
    if reference is not None:
        base = score(y, reference)
        point = {key: point[key] - base[key] for key in BOOT_METRICS}
    rng = np.random.default_rng(cfg['seed'])
    draws = {key: [] for key in BOOT_METRICS}
    for _ in range(cfg['bootstrap_repetitions']):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) != 2:
            continue
        result = score(y[index], probability[index])
        if reference is not None:
            base = score(y[index], reference[index])
            result = {key: result[key] - base[key] for key in BOOT_METRICS}
        for key in BOOT_METRICS:
            draws[key].append(result[key])
    if not all(draws.values()):
        raise ValueError('No valid two-class bootstrap samples')
    return [{'metric': key, 'estimate': point[key], 'lower_95': float(np.quantile(draws[key], .025)),
             'upper_95': float(np.quantile(draws[key], .975)), 'bootstrap_samples': len(draws[key])}
            for key in BOOT_METRICS]


def write_reports(predictions, cfg, summaries, report):
    aggregate, folds, intervals, paired, calibration = [], [], [], [], []
    for (experiment, model), group in predictions.groupby(['experiment', 'model'], sort=False):
        keys = {'experiment': experiment, 'cohort': EXPERIMENTS[experiment][0], 'model': model}
        aggregate.append({**keys, **all_metrics(group.label, group.probability, cfg['threshold'])})
        intervals.extend({**keys, **row} for row in bootstrap(group.label, group.probability, cfg))
        for fold, part in group.groupby('fold'):
            folds.append({**keys, 'fold': int(fold), **all_metrics(part.label, part.probability, cfg['threshold'])})
        observed, predicted = calibration_curve(group.label, group.probability, n_bins=5, strategy='quantile')
        calibration.extend({**keys, 'mean_probability': float(p), 'observed_fraction': float(o)} for p, o in zip(predicted, observed))
    for candidate, reference in COMPARISONS:
        if candidate not in set(predictions.experiment) or reference not in set(predictions.experiment):
            continue
        for model in cfg['models']:
            a = predictions[(predictions.experiment == candidate) & (predictions.model == model)]
            b = predictions[(predictions.experiment == reference) & (predictions.model == model)]
            joined = a.merge(b, on='RID', suffixes=('_candidate', '_reference'), validate='one_to_one')
            if (len(joined) != len(a) or len(joined) != len(b)
                    or not joined.label_candidate.eq(joined.label_reference).all()
                    or not joined.fold_candidate.eq(joined.fold_reference).all()):
                raise ValueError('Paired comparison participants, labels or folds differ')
            paired.extend({'candidate': candidate, 'reference': reference, 'model': model, **row}
                          for row in bootstrap(joined.label_candidate, joined.probability_candidate, cfg,
                                               reference=joined.probability_reference))
    results = pd.DataFrame(aggregate)
    for name, rows in [('metrics', aggregate), ('fold_metrics', folds), ('confidence_intervals', intervals),
                       ('paired_differences', paired), ('calibration', calibration)]:
        pd.DataFrame(rows).to_csv(report / (name + '.csv'), index=False)
    results.sort_values(['cohort', 'accuracy', 'auroc', 'brier_score'], ascending=[True, False, False, True]).to_csv(report / 'leaderboard.csv', index=False)
    for experiment, group in predictions.groupby('experiment', sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for model, part in group.groupby('model', sort=False):
            false_positive, true_positive, _ = roc_curve(part.label, part.probability)
            precision, recall, _ = precision_recall_curve(part.label, part.probability)
            observed, predicted = calibration_curve(part.label, part.probability, n_bins=5, strategy='quantile')
            axes[0].plot(false_positive, true_positive, label=model)
            axes[1].plot(recall, precision, label=model)
            axes[2].plot(predicted, observed, 'o-', label=model)
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[1].axhline(group.label.mean(), linestyle='--', color='gray')
        axes[2].plot([0, 1], [0, 1], '--', color='gray')
        for ax, title, xlabel, ylabel in zip(axes, ['ROC', 'Precision–recall', 'Calibration'],
                                           ['False positive rate', 'Recall', 'Mean predicted probability'],
                                           ['True positive rate', 'Precision', 'Observed AD fraction']):
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
            ax.legend(fontsize=7)
        fig.suptitle(experiment.replace('_', ' '))
        fig.tight_layout()
        fig.savefig(report / f'curves_{experiment}.png', dpi=160)
        plt.close(fig)
    for cohort, group in results.groupby('cohort', sort=False):
        fig, ax = plt.subplots(figsize=(10, 5))
        experiments = group.experiment.unique()
        width = .8 / len(experiments)
        centers = np.arange(len(cfg['models']))
        for i, experiment in enumerate(experiments):
            values = group[group.experiment == experiment].set_index('model').loc[cfg['models'], 'accuracy']
            ax.bar(centers + (i - (len(experiments) - 1) / 2) * width, 100 * values, width=width, label=experiment)
        ax.set(xticks=centers, xticklabels=cfg['models'], ylabel='Held-out accuracy (%)', ylim=(0, 100), title=cohort)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(report / f'accuracy_{cohort}.png', dpi=160)
        plt.close(fig)
    lines = ['# New modality comparison results', '', f"Policy: **{cfg['policy']}**; cohort set: **{cfg.get('cohort_set', 'mri_csf')}**.", '',
             'Five-fold pooled held-out predictions at threshold 0.5. Preprocessing is fitted inside each training fold.', '',
             '| Cohort | Participants | Converters | Non-converters |', '|---|---:|---:|---:|']
    lines += [f"| {row['cohort']} | {row['participants']} | {row['converters']} | {row['nonconverters']} |" for row in summaries]
    for cohort, group in results.groupby('cohort', sort=False):
        lines += ['', '## ' + cohort, '',
                  '| Experiment | Model | N | Accuracy | Precision | Sensitivity | Specificity | F1 | Balanced accuracy | AUROC | AP | Brier |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in group.to_dict('records'):
            rates = ' | '.join(f"{row[key]:.2%}" for key in ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'balanced_accuracy'])
            lines.append(f"| {row['experiment']} | {row['model']} | {row['participants']} | {rates} | {row['auroc']:.3f} | {row['average_precision']:.3f} | {row['brier_score']:.3f} |")
        lines += ['', f'![Accuracy comparison](accuracy_{cohort}.png)']
    lines += ['', '## Interpretation', '',
              '- Compare feature sets within a cohort: its participants and folds are identical. Different cohort sizes do not support direct claims of modality improvement.',
              '- All settings are fixed; no outer-fold threshold or hyperparameter optimization is performed.',
              '- These are exploratory internal comparisons on previously examined data, not external validation. Small sample sizes and complete-case selection limit generalization.',
              '- MRI/CSF collection is before baseline in the primary run. Later processing is permitted; clinical availability of those results at baseline is not established.',
              '- The same-day run is a separate boundary sensitivity; date-only records cannot establish ordering within baseline day.',
              '- Clinical/APOE predictors and provisional 36-month outcomes are frozen. The four-modality cohort remains empty. Different modality comparisons use separate eligible cohorts.',
              '- All CSV rates are proportions. AP is non-interpolated average precision; lower Brier is better.',
              '- Confidence intervals condition on saved predictions, omit full retraining uncertainty and are not adjusted for multiple comparisons.',
              '- paired_differences.csv reports candidate minus reference: positive accuracy/AUROC/AP and negative Brier favor the candidate.',
              '- SHAP is a separate optional step; explanations describe model behavior, not causal effects.', '']
    (report / 'results.md').write_text('\n'.join(lines), encoding='utf-8')


def main(config):
    config_path = local_path(config)
    cfg = json.loads(config_path.read_text())
    if cfg['folds'] != 5 or cfg['threshold'] != .5 or cfg['bootstrap_repetitions'] < 100:
        raise ValueError('Expected five folds, threshold 0.5 and at least 100 bootstrap draws')
    if not cfg['models'] or len(set(cfg['models'])) != len(cfg['models']) or not set(cfg['models']) <= {'logistic_regression', 'random_forest', 'xgboost'}:
        raise ValueError('Invalid model list')
    out, report = local_path(cfg['output_dir']), local_path(cfg['report_dir'])
    if not out.is_relative_to(ROOT / 'artifactsv2') or not report.is_relative_to(out):
        raise ValueError('Use artifactsv2/<run> and its reports subdirectory')
    if out == ROOT / 'artifactsv2' or report == out:
        raise ValueError('Separate run and reports subdirectories are required')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Run directory is not empty; choose a new output_dir/report_dir in a copied config')
    if 'xgboost' in cfg['models']:
        from xgboost import XGBClassifier
    cohorts, summaries, experiments, inputs = assemble(cfg)
    requested = cfg['explanations']['experiments']
    if not requested or not set(requested) <= set(experiments) or not set(cfg['explanations']['models']) <= set(cfg['models']):
        raise ValueError('Explanation selections must refer to trained experiments and models')
    inputs += [config_path, Path(__file__).resolve(), ROOT / 'src/scripts/train_models.py']
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True)
    (out / 'models').mkdir()
    packages = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'joblib'] + (['xgboost'] if 'xgboost' in cfg['models'] else [])
    manifest = {'status': 'running', 'config': cfg, 'python': platform.python_version(),
                'input_hashes': {str(path.relative_to(ROOT)): digest(path) for path in inputs},
                'versions': {package: importlib.metadata.version(package) for package in packages},
                'experiments': experiments, 'cohorts': summaries,
                'evaluation': 'Exploratory five-fold participant CV; feature comparisons share each cohort and its saved folds.'}
    save_json(out / 'manifest.json', manifest)
    try:
        save_json(report / 'cohort_summary.json', summaries)
        for name, frame in cohorts.items():
            frame.to_csv(out / f'{name}.csv', index_label='RID')
        warnings.filterwarnings('error', category=ConvergenceWarning)
        predictions = []
        for experiment, spec in experiments.items():
            frame, features = cohorts[spec['cohort']], spec['features']
            for model in cfg['models']:
                print(f'Training {experiment}: {model}, N={len(frame)}', flush=True)
                for fold in range(1, cfg['folds'] + 1):
                    train, held = frame[frame.fold != fold], frame[frame.fold == fold]
                    pipeline = make_pipeline(model, features, cfg)
                    pipeline.fit(train[features], train.label)
                    probability = pipeline.predict_proba(held[features])[:, 1]
                    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
                        raise ValueError('Invalid model probabilities')
                    joblib.dump(pipeline, out / 'models' / f'{experiment}__{model}__fold{fold}.joblib')
                    predictions.extend({'RID': rid, 'experiment': experiment, 'model': model, 'fold': fold,
                                        'label': int(label), 'probability': float(p), 'prediction': int(p >= cfg['threshold'])}
                                       for rid, label, p in zip(held.index, held.label, probability))
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(out / 'held_out_predictions.csv', index=False)
        print('Writing metrics, bootstrap intervals, matched comparisons and graphs.', flush=True)
        write_reports(predictions, cfg, summaries, report)
        if any(digest(local_path(path)) != expected for path, expected in manifest['input_hashes'].items()):
            raise ValueError('An input changed during training')
        manifest['status'] = 'complete'
        manifest['artifact_hashes'] = {str(path.relative_to(out)): digest(path) for path in out.rglob('*') if path.is_file() and path.name != 'manifest.json'}
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        save_json(out / 'manifest.json', manifest)
        raise
    save_json(out / 'manifest.json', manifest)
    print('Finished. Report: ' + str((report / 'results.md').relative_to(ROOT)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/modelingv2.json')
    main(parser.parse_args().config)
