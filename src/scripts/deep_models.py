"""User-run small tabular neural experiments; no fitting or downloads at import."""
import json
from dataset_paths import recorded_hash
from modeling_v2_inputs import load_v2, output_paths
import os
import platform
import random
import re
import warnings
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from train_models import ROOT, SEX, digest, local_path, make_pipeline, metrics, plt, read_indexed, save_json
from ensemble_models import paired_intervals


class MLP(nn.Module):
    def __init__(self, inputs, cfg):
        super().__init__()
        layers = []
        for width in cfg['hidden_sizes']:
            layers.extend([nn.Linear(inputs, width), nn.GELU(), nn.Dropout(cfg['dropout'])])
            inputs = width
        self.network = nn.Sequential(*layers, nn.Linear(inputs, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, width), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.network(x)


class TabularResNet(nn.Module):
    def __init__(self, inputs, cfg):
        super().__init__()
        width = cfg['width']
        self.network = nn.Sequential(
            nn.Linear(inputs, width),
            *[ResidualBlock(width, cfg['hidden_width'], cfg['dropout']) for _ in range(cfg['blocks'])],
            nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def device_for(cfg):
    name = cfg['device']
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name not in {'cpu', 'cuda'} or (name == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Use cpu, or cuda with an available CUDA GPU. auto uses CUDA or CPU.')
    return torch.device(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_data(cfg):
    folder = local_path(cfg['baseline_run'])
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or cfg['threshold'] != 0.5:
        raise ValueError('A complete baseline and threshold 0.5 are required')
    if (len(cfg['seeds']) < 2 or len(set(cfg['seeds'])) != len(cfg['seeds']) or
            any(not isinstance(seed, int) or seed < 0 for seed in cfg['seeds'])):
        raise ValueError('Provide at least two distinct nonnegative integer seeds')
    if cfg['inner_folds'] < 2 or cfg['bootstrap_repetitions'] < 100 or cfg['torch_threads'] < 1:
        raise ValueError('Invalid fold, bootstrap, or thread settings')
    if not cfg['experiments'] or len(set(cfg['experiments'])) != len(cfg['experiments']):
        raise ValueError('Provide distinct experiment names')
    training = cfg['training']
    if min(training[k] for k in ['learning_rate', 'batch_size', 'max_epochs', 'patience']) <= 0:
        raise ValueError('Learning rate, batch size, epochs, and patience must be positive')
    if cfg.get('input_format') == 'v2':
        return load_v2(cfg)
    schema_path = local_path(manifest['config']['input_dir']) / 'predictor_schema.json'
    if digest(schema_path) != recorded_hash(manifest['input_hashes'], schema_path):
        raise ValueError('The feature schema differs from the baseline run')
    allowed = set(json.loads(schema_path.read_text())['predictor_columns'])
    cohorts = {}
    for experiment in cfg['experiments']:
        if experiment not in {'clinical_reference', 'matched_clinical', 'matched_fusion'}:
            raise ValueError('Choose a documented clinical/fusion experiment')
        spec = manifest['experiments'][experiment]
        path = folder / (spec['cohort'] + '.csv')
        if digest(path) != manifest['artifact_hashes'][path.name]:
            raise ValueError('The saved cohort or folds changed since baseline')
        frame = read_indexed(path)
        features = spec['features']
        if not set(features) <= allowed or len(features) != len(set(features)):
            raise ValueError('Use only unique allowlisted features')
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError('This experiment requires complete finite predictors')
        if not frame.label.isin([0, 1]).all() or set(frame.fold) != set(range(1, 6)):
            raise ValueError('Expected the original binary labels and five saved outer folds')
        for fold in range(1, 6):
            train, held = frame[frame.fold != fold], frame[frame.fold == fold]
            if (train.label.value_counts().reindex([0, 1], fill_value=0).min() < cfg['inner_folds'] or
                    held.label.nunique() != 2):
                raise ValueError('Insufficient class counts for the requested folds')
        cohorts[experiment] = frame
    return manifest, cohorts


def train_network(x, y, kind, cfg, seed, device, epochs, validation=None):
    """Return a refit model or inner-fold best epoch, with no outer-fold access."""
    set_seed(seed)
    model = (MLP(x.shape[1], cfg['mlp']) if kind == 'mlp' else TabularResNet(x.shape[1], cfg['resnet'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                 weight_decay=cfg['training']['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    if validation is not None:
        vx = torch.as_tensor(np.asarray(validation[0], dtype=np.float32), device=device)
        vy = torch.as_tensor(np.asarray(validation[1], dtype=np.float32), device=device)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, stale = float('inf'), 1, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y), generator=generator).to(device)
        total = 0.0
        for indices in order.split(cfg['training']['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite neural training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
        row = {'epoch': epoch, 'train_loss': total / len(y)}
        if validation is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(vx), vy).cpu())
            if not np.isfinite(val_loss):
                raise ValueError('Nonfinite inner validation loss')
            row['validation_loss'] = val_loss
            if val_loss < best_loss - cfg['training']['min_delta']:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= cfg['training']['patience']:
            break
    model.eval()
    # Inner model weights are intentionally discarded. Final refits train for the selected fixed epoch count.
    return model, best_epoch if validation is not None else epochs, history


def predict_network(model, x, device):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(np.asarray(x, dtype=np.float32), device=device))).cpu().numpy()


def generate_reports(predictions, cfg, report, kind):
    rows, folds, paired = [], [], []
    primary = f'{kind}_seed{cfg["seeds"][0]}'
    mean_model = f'{kind}_seed_mean'
    for (experiment, name), group in predictions.groupby(['experiment', 'model'], sort=False):
        key = {'experiment': experiment, 'model': name}
        rows.append({**key, **metrics(group.label, group.probability, 0.5),
                     'accuracy': float(group.label.eq(group.prediction).mean())})
        for fold, part in group.groupby('fold'):
            folds.append({**key, 'fold': int(fold), **metrics(part.label, part.probability, 0.5),
                          'accuracy': float(part.label.eq(part.prediction).mean())})
        if name in {primary, mean_model}:
            baseline = predictions[(predictions.experiment == experiment) & (predictions.model == 'logistic_regression')]
            joined = group.merge(baseline, on='RID', suffixes=('_new', '_base'), validate='one_to_one')
            if (len(joined) != len(group) or len(joined) != len(baseline) or
                    not joined.fold_new.eq(joined.fold_base).all() or not joined.label_new.eq(joined.label_base).all()):
                raise ValueError('Paired comparison participant/fold/label mismatch')
            paired.extend({**key, 'reference': 'logistic_regression', **row} for row in
                          paired_intervals(joined.label_new, joined.probability_new, joined.probability_base, cfg))
    results = pd.DataFrame(rows)
    results.to_csv(report / 'metrics.csv', index=False)
    pd.DataFrame(folds).to_csv(report / 'fold_metrics.csv', index=False)
    pd.DataFrame(paired).to_csv(report / 'paired_vs_logistic.csv', index=False)
    seed_names = [f'{kind}_seed{seed}' for seed in cfg['seeds']]
    measures = ['accuracy', 'auroc', 'average_precision', 'brier_score', 'sensitivity', 'specificity']
    variability = []
    for experiment, group in results[results.model.isin(seed_names)].groupby('experiment'):
        for metric in measures:
            variability.append({'experiment': experiment, 'metric': metric, 'mean': group[metric].mean(),
                                'std': group[metric].std(ddof=1), 'min': group[metric].min(), 'max': group[metric].max()})
    pd.DataFrame(variability).to_csv(report / 'seed_variability.csv', index=False)
    for experiment, group in predictions.groupby('experiment'):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Ideal')
        for name in ['logistic_regression', primary, mean_model]:
            part = group[group.model == name]
            observed, predicted = calibration_curve(part.label, part.probability, n_bins=5, strategy='quantile')
            ax.plot(predicted, observed, 'o-', label=name)
        ax.set(xlabel='Mean held-out probability', ylabel='Observed AD fraction', title=experiment,
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(report / f'calibration_{experiment}.png', dpi=160)
        plt.close(fig)
    lines = [f'# Exploratory {kind} results', '',
             '| Experiment | Model | N | Accuracy | AUROC | AP | Brier |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['model']} | {row['participants']} | {row['accuracy']:.1%} | "
                     f"{row['auroc']:.3f} | {row['average_precision']:.3f} | {row['brier_score']:.3f} |")
    lines += ['', '## Interpretation', '',
              f'- The first configured seed ({cfg["seeds"][0]}) is the prespecified single-model result, not the best seed.',
              '- The seed_mean row averages probabilities from all configured seeds. It is a separate seed ensemble, not the mean of the accuracy scores.',
              '- Seed variability describes repeated fitting on the same participants. Seeds are not independent patient samples.',
              '- The fixed Logistic Regression reference is refitted in the same environment on identical outer folds.',
              '- Architecture, optimizer settings, and threshold were fixed. MLP/ResNet inner folds select training duration only; final models refit on the full outer training set.',
              '- TabPFN uses a pinned pretrained model and no task-specific gradient optimization; this is not training a transformer from scratch.',
              '- AP is non-interpolated average precision. Metric CSV rates use 0-1 scales; confusion counts are counts.',
              '- Paired intervals condition on saved predictions, omit full retraining uncertainty, and are not adjusted for multiple comparisons.',
              '- Previously inspected cohorts make this exploratory. Do not select the best result and call it independent confirmation.',
              '- Compare methods on identical saved participants and folds; cohort composition is recorded in the baseline manifest. Complete-case selection and provisional outcomes limit generalization.', '']
    (report / 'results.md').write_text('\n'.join(lines))
    return results


def run_deep_experiment(cfg, kind):
    if kind not in {'mlp', 'resnet', 'tabpfn'} or not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_name']):
        raise ValueError('Choose mlp/resnet/tabpfn and a simple run_name')
    original, cohorts = load_data(cfg)
    device = device_for(cfg)
    torch.set_num_threads(cfg['torch_threads'])
    checkpoint = None
    if kind == 'tabpfn':
        if version('tabpfn') != cfg['tabpfn']['package_version']:
            raise ValueError('Use the dedicated pinned TabPFN environment')
        checkpoint = local_path(cfg['tabpfn']['model_path'])
        if not checkpoint.is_file() or digest(checkpoint) != cfg['tabpfn']['checkpoint_sha256']:
            raise ValueError('Download the documented checkpoint in the notebook first; its SHA256 must match')
        # Fit/predict use the local checkpoint. Downloading weights is a separate explicit notebook cell.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
        from tabpfn import TabPFNClassifier
    out, report = output_paths(cfg, kind)
    for folder in [out, report]:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Outputs already exist. Skip fitting to read them, or change run_name for a new experiment.')
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    manifest = {'status': 'running', 'kind': kind, 'config': cfg, 'python': platform.python_version(),
                'device': str(device), 'baseline_config': original['config'], 'cohorts': original['cohorts'],
                'baseline_manifest_sha256': digest(local_path(cfg['baseline_run']) / 'manifest.json'),
                'source_hashes': {name: digest(ROOT / 'src/scripts' / name) for name in
                                  ['deep_models.py', 'ensemble_models.py', 'train_models.py']},
                'versions': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'joblib', 'matplotlib']},
                'experiment_specs': {name: original['experiments'][name] for name in cfg['experiments']},
                'primary_seed': cfg['seeds'][0], 'hyperparameter_search': False,
                'evaluation': 'Exploratory reuse of the saved original outer folds; no independent final test.'}
    if checkpoint is not None:
        manifest['versions']['tabpfn'] = version('tabpfn')
        manifest['checkpoint_sha256'] = digest(checkpoint)
        manifest['task_specific_gradient_training'] = False
    manifest['input_adapter_sha256'] = digest(ROOT / 'src/scripts/modeling_v2_inputs.py')
    save_json(out / 'manifest.json', manifest)
    predictions, epoch_rows, histories, split_rows = [], [], [], []

    def append_predictions(experiment, name, fold, held, probability):
        probability = np.asarray(probability)
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError('Invalid predicted probabilities')
        predictions.extend({'experiment': experiment, 'model': name, 'fold': fold, 'RID': rid,
                            'label': int(label), 'probability': float(p), 'prediction': int(p >= 0.5)}
                           for rid, label, p in zip(held.index, held.label, probability))

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            for experiment, frame in cohorts.items():
                features = original['experiments'][experiment]['features']
                frame[features + ['label', 'fold']].to_csv(out / f'{experiment}_cohort.csv', index_label='RID')
                for fold in range(1, 6):
                    train, held = frame[frame.fold != fold], frame[frame.fold == fold]
                    reference = make_pipeline('logistic_regression', features, original['config'])
                    reference.fit(train[features], train.label)
                    append_predictions(experiment, 'logistic_regression', fold, held,
                                       reference.predict_proba(held[features])[:, 1])
                    joblib.dump(reference, out / 'models' / f'{experiment}_fold{fold}_logistic.joblib')
                    seed_probabilities = []
                    for seed in cfg['seeds']:
                        print(f'{kind}: {experiment}, outer fold {fold}/5, seed {seed}', flush=True)
                        if kind == 'tabpfn':
                            set_seed(seed)
                            model = TabPFNClassifier(model_path=str(checkpoint), device=str(device),
                                                     n_estimators=cfg['tabpfn']['n_estimators'], random_state=seed,
                                                     categorical_features_indices=[features.index(SEX)] if SEX in features else None)
                            model.fit(train[features].to_numpy(), train.label.to_numpy())
                            probability = model.predict_proba(held[features].to_numpy())[:, 1]
                            # Recreate an in-context estimator from the cohort, checkpoint, version, and seed.
                            # Avoid serializing the large version-specific inference engine for every fold/seed.
                            save_json(out / 'models' / f'{experiment}_fold{fold}_seed{seed}_tabpfn.json', {
                                'seed': seed, 'outer_fold': fold, 'features': features,
                                'checkpoint': str(checkpoint.relative_to(ROOT)),
                                'checkpoint_sha256': cfg['tabpfn']['checkpoint_sha256'],
                                'n_estimators': cfg['tabpfn']['n_estimators'], 'package_version': version('tabpfn'),
                                'training_rows': f'{experiment}_cohort.csv, rows where fold != {fold}'})
                            del model
                        else:
                            best_epochs = []
                            inner = StratifiedKFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=cfg['seed'] + fold)
                            for inner_fold, (fitting, validation) in enumerate(inner.split(train, train.label), 1):
                                inner_train, inner_val = train.iloc[fitting], train.iloc[validation]
                                preprocessing = make_pipeline('logistic_regression', features, original['config']).named_steps['preprocessing']
                                x = preprocessing.fit_transform(inner_train[features])
                                vx = preprocessing.transform(inner_val[features])
                                inner_model, best_epoch, history = train_network(
                                    x, inner_train.label, kind, cfg, seed, device, cfg['training']['max_epochs'],
                                    validation=(vx, inner_val.label))
                                del inner_model
                                best_epochs.append(best_epoch)
                                context = {'experiment': experiment, 'outer_fold': fold, 'seed': seed, 'inner_fold': inner_fold}
                                histories.extend({**context, **row} for row in history)
                                split_rows.extend({**context, 'RID': rid} for rid in inner_val.index)
                            epochs = max(1, int(np.median(best_epochs)))
                            preprocessing = make_pipeline('logistic_regression', features, original['config']).named_steps['preprocessing']
                            x = preprocessing.fit_transform(train[features])
                            model, _, history = train_network(x, train.label, kind, cfg, seed, device, epochs)
                            probability = predict_network(model, preprocessing.transform(held[features]), device)
                            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            epoch_rows.append({'experiment': experiment, 'outer_fold': fold, 'seed': seed,
                                               'inner_best_epochs': json.dumps(best_epochs), 'refit_epochs': epochs,
                                               'encoded_input_columns': int(x.shape[1]), 'trainable_parameters': parameters})
                            histories.extend({'experiment': experiment, 'outer_fold': fold, 'seed': seed,
                                              'inner_fold': 0, **row} for row in history)
                            stem = f'{experiment}_fold{fold}_seed{seed}'
                            torch.save({'state_dict': {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                                        'kind': kind, 'architecture': cfg[kind], 'input_columns': int(x.shape[1]),
                                        'features': features, 'seed': seed, 'epochs': epochs}, out / 'models' / f'{stem}.pt')
                            joblib.dump(preprocessing, out / 'models' / f'{stem}_preprocessing.joblib')
                            del model
                        seed_probabilities.append(probability)
                        append_predictions(experiment, f'{kind}_seed{seed}', fold, held, probability)
                    append_predictions(experiment, f'{kind}_seed_mean', fold, held, np.mean(seed_probabilities, axis=0))
        prediction_frame = pd.DataFrame(predictions)
        if prediction_frame.duplicated(['experiment', 'model', 'RID']).any():
            raise ValueError('Expected exactly one prediction per participant per experiment/model')
        prediction_frame.to_csv(out / 'held_out_predictions.csv', index=False)
        if epoch_rows:
            pd.DataFrame(epoch_rows).to_csv(report / 'training_epochs.csv', index=False)
            pd.DataFrame(histories).to_csv(report / 'learning_curves.csv', index=False)
            pd.DataFrame(split_rows).to_csv(out / 'inner_validation_membership.csv', index=False)
        print('Writing metrics, seed variability, and paired comparisons.', flush=True)
        results = generate_reports(prediction_frame, cfg, report, kind)
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
