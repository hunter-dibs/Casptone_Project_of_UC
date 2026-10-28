"""User-run nonlinear feature representations with LR at both stacking levels."""
import argparse
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer, StandardScaler

from train_models import ROOT, SEX, digest, local_path, make_pipeline, read_indexed, save_json
from ensemble_models import paired_intervals, report_results

VIEWS = ['linear', 'spline', 'interactions', 'rbf']


def load_cohorts(cfg):
    baseline = local_path(cfg['baseline_run'])
    manifest = json.loads((baseline / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or cfg['threshold'] != 0.5:
        raise ValueError('Use a completed original baseline and threshold 0.5')
    if min(cfg['inner_folds'], cfg['stacking_folds']) < 2 or cfg['bootstrap_repetitions'] < 100:
        raise ValueError('At least two inner/stacking folds and 100 bootstrap draws are required')
    if not cfg['experiments'] or len(set(cfg['experiments'])) != len(cfg['experiments']):
        raise ValueError('Provide distinct experiment names')
    for name in ['meta_C_candidates', 'baseline_C_candidates']:
        if not cfg[name] or any(c <= 0 for c in cfg[name]):
            raise ValueError('Regularization C values must be positive')
    if set(cfg['view_C']) != set(VIEWS) or any(c <= 0 for c in cfg['view_C'].values()):
        raise ValueError('Specify positive C for the four feature representations')
    if cfg.get('input_format') == 'v2':
        return load_v2(cfg)
    schema_path = local_path(manifest['config']['input_dir']) / 'predictor_schema.json'
    if digest(schema_path) != recorded_hash(manifest['input_hashes'], schema_path):
        raise ValueError('The predictor schema changed since the original run')
    allowed = set(json.loads(schema_path.read_text())['predictor_columns'])
    cohorts = {}
    for experiment in cfg['experiments']:
        if experiment not in {'clinical_reference', 'matched_clinical', 'matched_fusion'}:
            raise ValueError('Choose a documented clinical/fusion cohort')
        spec = manifest['experiments'][experiment]
        path = baseline / (spec['cohort'] + '.csv')
        if digest(path) != manifest['artifact_hashes'][path.name]:
            raise ValueError('Saved cohort or folds changed')
        features = spec['features']
        if len(features) != len(set(features)) or not set(features) <= allowed:
            raise ValueError('Use the saved unique allowlisted features')
        frame = read_indexed(path)
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError('This experiment requires complete finite predictors')
        if not frame.label.isin([0, 1]).all() or set(frame.fold) != set(range(1, 6)):
            raise ValueError('Expected the original five outer folds and binary outcomes')
        for fold in range(1, 6):
            outer_train = frame[frame.fold != fold]
            for fitting, _ in StratifiedKFold(cfg['inner_folds'], shuffle=True, random_state=cfg['seed'] + fold).split(outer_train, outer_train.label):
                counts = outer_train.iloc[fitting].label.value_counts().reindex([0, 1], fill_value=0)
                if counts.min() < cfg['stacking_folds']:
                    raise ValueError('Too few members of a class for the nested stacking splits')
        cohorts[experiment] = frame
    return manifest, cohorts


def learner(view, features, cfg):
    numeric = [name for name in features if name != SEX]
    # Constant-column selection is fitted inside the current training partition.
    numeric_steps = [('nonconstant', VarianceThreshold()), ('scale', StandardScaler())]
    if view == 'spline':
        numeric_steps.extend([
            ('splines', SplineTransformer(n_knots=cfg['spline_knots'], degree=cfg['spline_degree'],
                                         knots='uniform', extrapolation='constant', include_bias=False)),
            ('basis_scale', StandardScaler())])
    elif view == 'interactions':
        numeric_steps.extend([
            ('pairs', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('interaction_scale', StandardScaler())])
    elif view not in {'linear', 'rbf'}:
        raise ValueError('Unknown feature representation')
    transformers = [('numeric', Pipeline(numeric_steps), numeric)]
    if SEX in features:
        transformers.append(('sex', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [SEX]))
    steps = [('preprocessing', ColumnTransformer(transformers, remainder='drop'))]
    if view == 'rbf':
        steps.extend([
            ('kernel_features', RBFSampler(gamma=cfg['rbf_gamma'], n_components=cfg['rbf_components'], random_state=cfg['seed'])),
            ('kernel_scale', StandardScaler())])
    steps.append(('model', LogisticRegression(C=cfg['view_C'][view], max_iter=cfg['max_iter'],
                                             solver='lbfgs', random_state=cfg['seed'])))
    return Pipeline(steps)


def make_stack(features, cfg, fold):
    return StackingClassifier(
        estimators=[(name, learner(name, features, cfg)) for name in VIEWS],
        final_estimator=LogisticRegression(C=1.0, max_iter=cfg['max_iter'], solver='lbfgs', random_state=cfg['seed']),
        stack_method='predict_proba', passthrough=False, n_jobs=1,
        cv=StratifiedKFold(cfg['stacking_folds'], shuffle=True, random_state=cfg['seed'] + 100 + fold))


def select_by_accuracy(cv_results):
    """Predeclared: highest mean inner accuracy; ties favor lower inner log loss."""
    accuracy = np.asarray(cv_results['mean_test_accuracy'])
    negative_loss = np.asarray(cv_results['mean_test_neg_log_loss'])
    if not np.isfinite(accuracy).all() or not np.isfinite(negative_loss).all():
        raise ValueError('An inner candidate has nonfinite scores')
    return int(np.lexsort((-negative_loss, -accuracy))[0])


def accuracy_at_half(y, probability):
    return float(np.mean(np.asarray(y) == (np.asarray(probability) >= 0.5)))


def search(estimator, grid, splits):
    scoring = {'accuracy': make_scorer(accuracy_at_half, response_method='predict_proba'), 'neg_log_loss': 'neg_log_loss'}
    return GridSearchCV(estimator, grid, cv=splits, scoring=scoring,
                        refit=select_by_accuracy, n_jobs=1, error_score='raise', return_train_score=False)


def write_summary(results, report):
    lines = ['# Exploratory Logistic Regression stacking results', '',
             '| Experiment | Model | N | Accuracy | AUROC | AP | Brier |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in results.to_dict('records'):
        lines.append(f"| {row['experiment']} | {row['model']} | {row['participants']} | {row['accuracy']:.1%} | "
                     f"{row['auroc']:.3f} | {row['average_precision']:.3f} | {row['brier_score']:.3f} |")
    lines += ['', '## Interpretation', '',
              '- The primary new method is lr_stacking. All four base models and the final combiner are regularized Logistic Regression.',
              '- The base representations are linear, additive splines, pairwise numeric interactions, and approximate RBF kernel features.',
              '- Hyperparameter search refits the entire stacking estimator inside each inner training split. Its own deeper cross-fitting trains the combiner on held-out base probabilities.',
              '- Preprocessing is inside every base pipeline, including the deepest split. No globally precomputed transforms or outer outcomes enter training.',
              '- Inner selection maximizes accuracy at threshold 0.5, with log loss as tie-breaker. The same selection rule is used for the tuned plain LR reference.',
              '- Only combiner C is searched for the stack; base representation settings are prespecified. Search budgets are bounded, not equivalent in computational cost.',
              '- Component rows and equal voting are descriptive comparisons, not separately selected winners. Meta coefficients are not causal effects or feature explanations.',
              '- AP is non-interpolated average precision. CSV rates use the 0-1 scale; confusion entries are participant counts.',
              '- Paired intervals compare against fixed and tuned LR, condition on predictions, omit full retraining uncertainty, and are not adjusted for multiple comparisons.',
              '- These are exploratory comparisons using previously inspected cohorts. No external validation or guaranteed accuracy improvement is established.',
              '- Compare methods on identical saved participants and folds; cohort composition is recorded in the baseline manifest. Complete-case selection and provisional outcomes limit generalization.', '']
    (report / 'results.md').write_text('\n'.join(lines))


def run_lr_stacking(cfg):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_name']):
        raise ValueError('Use a simple run_name')
    original, cohorts = load_cohorts(cfg)
    out, report = output_paths(cfg)
    for folder in [out, report]:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Output directory already contains a run; choose a new run_name or skip fitting')
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    manifest = {'status': 'running', 'config': cfg, 'python': platform.python_version(),
                'original_config': original['config'], 'cohorts': original['cohorts'],
                'experiment_specs': {name: original['experiments'][name] for name in cfg['experiments']},
                'baseline_manifest_sha256': digest(local_path(cfg['baseline_run']) / 'manifest.json'),
                'source_hashes': {name: digest(ROOT / 'src/scripts' / name) for name in
                                  ['lr_stacking.py', 'train_models.py', 'ensemble_models.py']},
                'versions': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'matplotlib']},
                'primary_method': 'lr_stacking',
                'selection': 'Mean inner accuracy, negative log loss as tie-breaker; entire stack nested within search.',
                'evaluation': 'Exploratory reuse of original saved outer folds.'}
    manifest['input_adapter_sha256'] = digest(ROOT / 'src/scripts/modeling_v2_inputs.py')
    save_json(out / 'manifest.json', manifest)
    predictions, searches, selected, coefficients, membership = [], [], [], [], []
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            for experiment, frame in cohorts.items():
                features = original['experiments'][experiment]['features']
                frame[features + ['label', 'fold']].to_csv(out / f'{experiment}_cohort.csv', index_label='RID')
                for fold in range(1, 6):
                    print(f'LR stacking: {experiment}, outer fold {fold}/5', flush=True)
                    train, held = frame[frame.fold != fold], frame[frame.fold == fold]
                    splits = list(StratifiedKFold(cfg['inner_folds'], shuffle=True, random_state=cfg['seed'] + fold).split(train, train.label))
                    for inner_fold, (fitting, validation) in enumerate(splits, 1):
                        membership.extend({'experiment': experiment, 'outer_fold': fold, 'level': 'search',
                                           'inner_fold': inner_fold, 'stack_fold': 0, 'RID': rid} for rid in train.iloc[validation].index)
                        sub = train.iloc[fitting]
                        deeper = StratifiedKFold(cfg['stacking_folds'], shuffle=True, random_state=cfg['seed'] + 100 + fold)
                        for stack_fold, (_, deeper_val) in enumerate(deeper.split(sub, sub.label), 1):
                            membership.extend({'experiment': experiment, 'outer_fold': fold, 'level': 'stack_within_search',
                                               'inner_fold': inner_fold, 'stack_fold': stack_fold, 'RID': rid} for rid in sub.iloc[deeper_val].index)
                    final_splits = StratifiedKFold(cfg['stacking_folds'], shuffle=True, random_state=cfg['seed'] + 100 + fold)
                    for stack_fold, (_, validation) in enumerate(final_splits.split(train, train.label), 1):
                        membership.extend({'experiment': experiment, 'outer_fold': fold, 'level': 'stack_final_refit',
                                           'inner_fold': 0, 'stack_fold': stack_fold, 'RID': rid} for rid in train.iloc[validation].index)

                    fixed = make_pipeline('logistic_regression', features, original['config'])
                    fixed.fit(train[features], train.label)
                    plain = make_pipeline('logistic_regression', features, original['config'])
                    plain.set_params(model__max_iter=cfg['max_iter'])
                    plain_search = search(plain, {'model__C': cfg['baseline_C_candidates']}, splits)
                    plain_search.fit(train[features], train.label)
                    stack_search = search(make_stack(features, cfg, fold),
                                          {'final_estimator__C': cfg['meta_C_candidates']}, splits)
                    stack_search.fit(train[features], train.label)
                    stack = stack_search.best_estimator_
                    for family, fitted_search in [('tuned_logistic', plain_search), ('lr_stacking', stack_search)]:
                        result = fitted_search.cv_results_
                        for i, params in enumerate(result['params']):
                            searches.append({'experiment': experiment, 'outer_fold': fold, 'family': family,
                                             'parameters': json.dumps(params), 'inner_accuracy': float(result['mean_test_accuracy'][i]),
                                             'inner_log_loss': float(-result['mean_test_neg_log_loss'][i]),
                                             'selected': i == fitted_search.best_index_})
                    selected.append({'experiment': experiment, 'outer_fold': fold,
                                     'baseline_C': plain_search.best_params_['model__C'],
                                     'meta_C': stack_search.best_params_['final_estimator__C']})
                    coefficients.append({'experiment': experiment, 'outer_fold': fold,
                                         'intercept': float(stack.final_estimator_.intercept_[0]),
                                         **dict(zip(VIEWS, stack.final_estimator_.coef_[0].tolist()))})
                    models = {'logistic_regression': fixed, 'tuned_logistic_regression': plain_search.best_estimator_,
                              'lr_stacking': stack}
                    for name, model in models.items():
                        joblib.dump(model, out / 'models' / f'{experiment}_fold{fold}_{name}.joblib')
                    probabilities = {name: model.predict_proba(held[features])[:, 1] for name, model in models.items()}
                    base_probabilities = []
                    for view in VIEWS:
                        p = stack.named_estimators_[view].predict_proba(held[features])[:, 1]
                        probabilities[f'lr_{view}_view'] = p
                        base_probabilities.append(p)
                    probabilities['lr_equal_voting'] = np.mean(base_probabilities, axis=0)
                    for name, p in probabilities.items():
                        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
                            raise ValueError('Invalid predicted probabilities')
                        predictions.extend({'experiment': experiment, 'model': name, 'fold': fold, 'RID': rid,
                                            'label': int(label), 'probability': float(value), 'prediction': int(value >= 0.5)}
                                           for rid, label, value in zip(held.index, held.label, p))
        predictions = pd.DataFrame(predictions)
        if predictions.duplicated(['experiment', 'model', 'RID']).any():
            raise ValueError('Duplicate held-out predictions')
        predictions.to_csv(out / 'held_out_predictions.csv', index=False)
        pd.DataFrame(membership).to_csv(out / 'inner_membership.csv', index=False)
        pd.DataFrame(searches).to_csv(report / 'search_results.csv', index=False)
        pd.DataFrame(selected).to_csv(report / 'selected_parameters.csv', index=False)
        pd.DataFrame(coefficients).to_csv(report / 'meta_coefficients.csv', index=False)
        results = report_results(predictions, cfg, report, ['lr_stacking', 'lr_equal_voting'])
        write_summary(results, report)
        paired = []
        for experiment in cfg['experiments']:
            reference = predictions[(predictions.experiment == experiment) &
                                    (predictions.model == 'tuned_logistic_regression')].set_index('RID')
            for name in ['lr_stacking', 'lr_equal_voting']:
                group = predictions[(predictions.experiment == experiment) & (predictions.model == name)].set_index('RID')
                aligned = reference.reindex(group.index)
                if len(reference) != len(group) or not aligned.label.equals(group.label) or not aligned.fold.equals(group.fold):
                    raise ValueError('Tuned reference comparison mismatch')
                paired.extend({'experiment': experiment, 'model': name, 'reference': 'tuned_logistic_regression', **row}
                              for row in paired_intervals(group.label, group.probability, aligned.probability, cfg))
        pd.DataFrame(paired).to_csv(report / 'paired_vs_tuned_logistic.csv', index=False)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/lr_stacking.json')
    args = parser.parse_args()
    run_lr_stacking(json.loads(local_path(args.config).read_text()))
