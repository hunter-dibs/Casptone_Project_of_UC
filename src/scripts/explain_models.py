#!/usr/bin/env python3
"""Explain saved held-out predictions after a successful training run."""
import argparse
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import shap

from train_models import ROOT, digest, local_path, plt, save_json


def bar_plot(values, title, xlabel, path):
    values = values.sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, len(values) * 0.32)))
    ax.barh(values.index, values.values, color=['#32689c' if v >= 0 else '#b45c48' for v in values.values])
    ax.set(title=title, xlabel=xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(config='config/modeling.json', experiment_override=None):
    cfg = json.loads(local_path(config).read_text())
    out, report = local_path(cfg['output_dir']), local_path(cfg['report_dir'])
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or cfg != manifest['config']:
        raise ValueError('Explanations require a completed run and its unchanged configuration')
    settings = cfg['explanations']
    configured = settings.get('experiments', [settings.get('experiment')])
    if not configured or any(name not in manifest['experiments'] for name in configured):
        raise ValueError('Explanation experiments must be present in the completed run')
    if experiment_override is None and 'experiments' in settings:
        for experiment in configured:
            main(config, experiment)
        return
    experiment = experiment_override or configured[0]
    if experiment not in configured:
        raise ValueError('Requested explanation experiment is not configured')
    spec = manifest['experiments'][experiment]
    features = spec['features']
    names = settings['models']
    if not names or not set(names) <= set(cfg['models']):
        raise ValueError('Explanation models must be among the trained models')
    if min(settings['background_participants'], settings['permutation_cycles'], settings['top_features']) < 1:
        raise ValueError('Explanation settings must be positive')
    # Only load artifacts produced by this training run. Never load third-party joblib files.
    required = [f"{spec['cohort']}.csv", 'held_out_predictions.csv']
    required += [f'models/{experiment}__{name}__fold{fold}.joblib'
                 for name in names for fold in range(1, cfg['folds'] + 1)]
    for name in required:
        if digest(out / name) != manifest['artifact_hashes'][name]:
            raise ValueError(f'Saved training artifact changed: {name}')
    frame = pd.read_csv(out / f"{spec['cohort']}.csv", dtype={'RID': str}).set_index('RID')
    predictions = pd.read_csv(out / 'held_out_predictions.csv', dtype={'RID': str})
    detail_dir, figure_dir = out / 'explanations', report / 'explanations'
    if 'experiments' in settings:
        detail_dir, figure_dir = detail_dir / experiment, figure_dir / experiment
    for folder in [detail_dir, figure_dir]:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Explanation outputs already exist; use a fresh run directory')
    detail_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    ranking_rows, stability, example_rows = [], [], []
    for name in names:
        print(f'Explaining {experiment}: {name}', flush=True)
        explained = []
        for fold in range(1, cfg['folds'] + 1):
            train, held = frame[frame.fold != fold], frame[frame.fold == fold]
            background = train[features].sample(n=min(settings['background_participants'], len(train)),
                                                random_state=cfg['seed'] + fold)
            pipeline = joblib.load(out / 'models' / f'{experiment}__{name}__fold{fold}.joblib')

            def probability(values):
                return pipeline.predict_proba(pd.DataFrame(values, columns=features))[:, 1]

            # A shared probability scale permits comparisons across model families.
            explainer = shap.Explainer(probability, background.to_numpy(), algorithm='permutation',
                                       feature_names=features, seed=cfg['seed'] + fold)
            explanation = explainer(held[features].to_numpy(),
                                    max_evals=(2 * len(features) + 1) * settings['permutation_cycles'],
                                    silent=True)
            expected = predictions[(predictions.experiment == experiment) &
                                   (predictions.model == name)].set_index('RID').loc[held.index, 'probability']
            reconstructed = np.asarray(explanation.base_values).reshape(-1) + explanation.values.sum(axis=1)
            if not np.allclose(reconstructed, expected.to_numpy(), atol=1e-5, rtol=1e-5):
                raise ValueError('SHAP contributions do not reconstruct the saved held-out predictions')
            values = pd.DataFrame(explanation.values, index=held.index, columns=features)
            values['base_probability'] = np.asarray(explanation.base_values).reshape(-1)
            values['probability'] = expected
            values['label'] = held.label
            values['fold'] = fold
            explained.append(values)
            importance = values[features].abs().mean().sort_values(ascending=False)
            ranking_rows.extend({'model': name, 'fold': fold, 'feature': feature,
                                 'mean_abs_shap': float(value), 'rank': rank}
                                for rank, (feature, value) in enumerate(importance.items(), 1))
        values = pd.concat(explained).sort_index()
        values.to_csv(detail_dir / f'{name}_held_out_shap.csv', index_label='RID')
        importance = values[features].abs().mean().sort_values(ascending=False)
        importance.rename('mean_abs_shap').to_csv(figure_dir / f'{name}_global_importance.csv', index_label='feature')
        bar_plot(importance.head(15), f'{name}: held-out global importance',
                 'Mean absolute contribution to predicted probability', figure_dir / f'{name}_global.png')
        # Deterministic examples: first string RID in each observed class, without choosing by model accuracy.
        for label in [0, 1]:
            example = values[values.label == label].iloc[0]
            rid = example.name
            contributions = example[features].astype(float)
            selected = contributions.abs().nlargest(10).index
            bar_plot(contributions[selected], f'{name}: example with outcome {label}',
                     'Contribution to predicted probability', figure_dir / f'{name}_local_class{label}.png')
            example_rows.append({'model': name, 'label': label, 'RID': rid,
                                 'base_probability': example.base_probability,
                                 'probability': example.probability,
                                 'omitted_contribution_sum': float(contributions.drop(selected).sum())})
        ranks = pd.DataFrame([row for row in ranking_rows if row['model'] == name])
        for first, second in itertools.combinations(range(1, cfg['folds'] + 1), 2):
            a = ranks[ranks.fold == first].set_index('feature')['rank'].reindex(features)
            b = ranks[ranks.fold == second].set_index('feature')['rank'].reindex(features)
            top_a, top_b = set(a.nsmallest(settings['top_features']).index), set(b.nsmallest(settings['top_features']).index)
            stability.append({'model': name, 'fold_a': first, 'fold_b': second,
                              'rank_correlation': float(a.corr(b)),
                              'top_feature_jaccard': len(top_a & top_b) / len(top_a | top_b)})
    pd.DataFrame(ranking_rows).to_csv(figure_dir / 'fold_feature_rankings.csv', index=False)
    pd.DataFrame(stability).to_csv(figure_dir / 'fold_stability.csv', index=False)
    pd.DataFrame(example_rows).to_csv(detail_dir / 'local_examples.csv', index=False)
    save_json(detail_dir / 'manifest.json', {
        'status': 'complete', 'settings': settings, 'experiment': experiment, 'shap_version': shap.__version__,
        'training_manifest_sha256': digest(out / 'manifest.json'),
        'script_sha256': digest(ROOT / 'src/scripts' / 'explain_models.py'),
        'scale': 'positive-class probability',
        'local_example_rule': 'First string RID in each observed class; full identifiers saved locally only.'})
    (figure_dir / 'README.md').write_text(
        '# Held-out SHAP explanations\n\n'
        'Permutation SHAP explains predicted AD probability using training-fold backgrounds only. '
        'All held-out participants in the selected experiment are explained. '
        'Global values are mean absolute contributions; local plots show the ten largest absolute contributions. '
        'Base probabilities, omitted contribution sums, and participant-level values are in the local artifacts directory.\n\n'
        'Local examples are the first string RID in each observed class, selected independently of model accuracy. '
        'Fold stability reports rank correlation and top-feature Jaccard overlap. These are descriptive: '
        'fold models share training participants and are not independent replications. '
        'Correlated cognitive features can share or exchange importance; marginal permutation may create unusual feature combinations. '
        'The explanations describe model behavior, not causal effects or clinical validity.\n')
    print(f'Finished. Explanation figures: {figure_dir.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/modeling.json')
    parser.add_argument('--experiment', help='Explain only this configured experiment (otherwise all configured experiments)')
    args = parser.parse_args()
    main(args.config, args.experiment)
