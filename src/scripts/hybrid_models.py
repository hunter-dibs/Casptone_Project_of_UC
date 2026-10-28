"""User-run anchored tabular attention experiment. No automatic execution."""
import json
from modeling_v2_inputs import output_paths
import platform
import re
import warnings
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from train_models import ROOT, digest, local_path, make_pipeline, save_json
from deep_models import load_data, device_for, set_seed, generate_reports, ResidualBlock
from ensemble_models import paired_intervals


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4),
                                         nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width),
                                         nn.Dropout(dropout))

    def forward(self, x):
        z = self.norm_attention(x)
        attended, _ = self.attention(z, z, z, need_weights=False)
        x = x + self.dropout(attended)
        return x + self.feedforward(x)


class AnchoredAttention(nn.Module):
    """Frozen training-only LR logit plus gated attention and residual-MLP corrections."""
    def __init__(self, inputs, candidate, coefficient, intercept):
        super().__init__()
        width, dropout = candidate['token_width'], candidate['dropout']
        self.register_buffer('anchor_coefficient', torch.as_tensor(coefficient, dtype=torch.float32).reshape(-1))
        self.register_buffer('anchor_intercept', torch.as_tensor(float(intercept), dtype=torch.float32))
        # Each encoded column has its own learned identity and value embedding.
        # There is no spatial neighborhood or image reshape.
        self.token_weight = nn.Parameter(torch.empty(inputs, width))
        self.token_bias = nn.Parameter(torch.empty(inputs, width))
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.token_weight, std=0.02)
        nn.init.normal_(self.token_bias, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.attention = nn.Sequential(*[
            AttentionBlock(width, candidate['heads'], dropout)
            for _ in range(candidate['attention_blocks'])])
        self.attention_norm = nn.LayerNorm(width)
        self.attention_head = nn.Linear(width, 1)
        residual_width = candidate['residual_width']
        self.residual = nn.Sequential(nn.Linear(inputs, residual_width), nn.GELU(), *[
            ResidualBlock(residual_width, 2 * residual_width, dropout)
            for _ in range(candidate['residual_blocks'])], nn.LayerNorm(residual_width))
        self.residual_head = nn.Linear(residual_width, 1)
        self.gates = nn.Parameter(torch.full((2,), -2.0))
        # Epoch zero reproduces the LR anchor; learning must earn a correction.
        for head in [self.attention_head, self.residual_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def components(self, x):
        tokens = x.unsqueeze(-1) * self.token_weight.unsqueeze(0) + self.token_bias.unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(len(x), -1, -1), tokens], dim=1)
        attended = self.attention(tokens)[:, 0]
        attention_logit = self.attention_head(self.attention_norm(attended)).squeeze(-1)
        residual_logit = self.residual_head(self.residual(x)).squeeze(-1)
        gates = torch.sigmoid(self.gates)
        correction = gates[0] * attention_logit + gates[1] * residual_logit
        anchor = x @ self.anchor_coefficient + self.anchor_intercept
        return anchor + correction, correction

    def forward(self, x):
        return self.components(x)[0]


def fit_network(x, y, candidate, anchor, cfg, seed, device, epochs, validation=None):
    set_seed(seed)
    model = AnchoredAttention(x.shape[1], candidate, anchor.coef_[0], anchor.intercept_[0]).to(device)
    decayed = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    undecayed = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {'params': decayed, 'weight_decay': candidate['weight_decay']},
        {'params': undecayed, 'weight_decay': 0.0}], lr=candidate['learning_rate'])
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    model.eval()
    with torch.no_grad():
        start_loss = float(loss_fn(model(x), y).cpu())
    history = [{'epoch': 0, 'train_bce': start_loss}]
    best_epoch, stale = 0, 0
    if validation is not None:
        vx = torch.as_tensor(np.asarray(validation[0], dtype=np.float32), device=device)
        vy = torch.as_tensor(np.asarray(validation[1], dtype=np.float32), device=device)
        with torch.no_grad():
            best_loss = float(loss_fn(model(vx), vy).cpu())
        history[0]['validation_bce'] = best_loss
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        order = torch.randperm(len(y), generator=generator).to(device)
        for indices in order.split(cfg['training']['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            logits, correction = model.components(x[indices])
            bce = loss_fn(logits, y[indices])
            loss = bce + cfg['training']['correction_penalty'] * correction.square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite hybrid loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'])
            optimizer.step()
            total += float(bce.detach().cpu()) * len(indices)
        row = {'epoch': epoch, 'train_bce': total / len(y)}
        if validation is not None:
            model.eval()
            with torch.no_grad():
                current = float(loss_fn(model(vx), vy).cpu())
            if not np.isfinite(current):
                raise ValueError('Nonfinite inner validation loss')
            row['validation_bce'] = current
            if current < best_loss - cfg['training']['min_delta']:
                best_loss, best_epoch, stale = current, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= cfg['training']['patience']:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch if validation is not None else epochs, history


def predict(model, x, device):
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        return torch.sigmoid(model(x)).cpu().numpy()


def write_summary(results, cfg, report):
    lines = ['# Exploratory anchored-attention results', '',
             '**90% accuracy is a target, not an assumed outcome.** All rows below use held-out predictions at threshold 0.5.', '',
             '| Experiment | Model | N | Accuracy | AUROC | AP | Brier | Reached 90%? |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for row in results.to_dict('records'):
        lines.append(f"| {row['experiment']} | {row['model']} | {row['participants']} | {row['accuracy']:.1%} | "
                     f"{row['auroc']:.3f} | {row['average_precision']:.3f} | {row['brier_score']:.3f} | "
                     f"{'Yes' if row['accuracy'] >= cfg['target_accuracy'] else 'No'} |")
    lines += ['', '## Protocol and interpretation', '',
              '- Original cohorts, labels, features, and five outer folds are unchanged. These already-inspected data support exploratory conclusions only.',
              '- The frozen LR anchor is fitted only on the relevant training partition. Attention and residual-MLP branches learn gated logit corrections.',
              '- Two neural configurations are selected by inner-fold log loss using the configured tuning seed. Inner validation also selects stopping epochs; the median chosen epoch is used for full outer-training refits.',
              '- Epoch zero is eligible. If corrections do not help inner validation, the refit can reduce to the LR anchor. This is reported explicitly, not presented as a neural gain.',
              '- Seeds 42/43/44 are all reported; the first configured seed is primary. seed_mean averages probabilities and is a separate ensemble.',
              '- Tuned Logistic Regression gets its own training-only C search. Compare against it as well as the original fixed-C reference.',
              '- Accuracy is not optimized by changing the outer threshold, dropping difficult participants, or inspecting outer labels during tuning.',
              '- AP is non-interpolated average precision; lower Brier is better. Check sensitivity, specificity, and calibration alongside accuracy.',
              '- Paired bootstrap intervals condition on saved predictions, omit full retraining uncertainty, and are not corrected for multiple comparisons.',
              '- A point estimate above 90% on these data would not establish external performance or clinical readiness. Do not choose the best seed post hoc.',
              '- Compare methods on identical saved participants and folds; cohort composition is recorded in the baseline manifest. Complete-case selection and provisional outcomes limit generalization.', '']
    (report / 'results.md').write_text('\n'.join(lines))


def run_hybrid(cfg):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_name']):
        raise ValueError('Use a simple run_name')
    if not cfg['candidates'] or not cfg['lr_C_candidates'] or any(c <= 0 for c in cfg['lr_C_candidates']):
        raise ValueError('Provide candidate architectures and positive LR C values')
    if len({c['name'] for c in cfg['candidates']}) != len(cfg['candidates']):
        raise ValueError('Candidate names must be distinct')
    for candidate in cfg['candidates']:
        if candidate['token_width'] % candidate['heads'] or not 0 <= candidate['dropout'] < 1:
            raise ValueError('Invalid attention head count or dropout')
    original, cohorts = load_data(cfg)
    device = device_for(cfg)
    torch.set_num_threads(cfg['torch_threads'])
    out, report = output_paths(cfg)
    for folder in [out, report]:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Outputs already exist; read them or choose a new run_name')
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    manifest = {'status': 'running', 'config': cfg, 'python': platform.python_version(), 'device': str(device),
                'baseline_config': original['config'], 'cohorts': original['cohorts'],
                'baseline_manifest_sha256': digest(local_path(cfg['baseline_run']) / 'manifest.json'),
                'source_hashes': {name: digest(ROOT / 'src/scripts' / name) for name in
                                  ['hybrid_models.py', 'deep_models.py', 'train_models.py', 'ensemble_models.py']},
                'versions': {name: version(name) for name in ['torch', 'numpy', 'pandas', 'scikit-learn', 'joblib', 'matplotlib']},
                'primary_seed': cfg['seeds'][0], 'selection_metric': 'inner_log_loss',
                'evaluation': 'Exploratory reuse of original outer folds; target is not a promised result.'}
    manifest['input_adapter_sha256'] = digest(ROOT / 'src/scripts/modeling_v2_inputs.py')
    save_json(out / 'manifest.json', manifest)
    predictions, search_rows, selection_rows, learning, inner_membership = [], [], [], [], []

    def append(experiment, name, fold, held, probability):
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError('Invalid probabilities')
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
                    print(f'Hybrid attention: {experiment}, outer fold {fold}/5', flush=True)
                    train, held = frame[frame.fold != fold], frame[frame.fold == fold]
                    splits = list(StratifiedKFold(n_splits=cfg['inner_folds'], shuffle=True,
                                                 random_state=cfg['seed'] + fold).split(train, train.label))
                    candidate_predictions = {c['name']: np.zeros(len(train)) for c in cfg['candidates']}
                    epochs_by_candidate = {c['name']: [] for c in cfg['candidates']}
                    lr_predictions = {c: np.zeros(len(train)) for c in cfg['lr_C_candidates']}
                    inner_fold_ids = np.zeros(len(train), dtype=int)
                    for inner_fold, (fitting, validation) in enumerate(splits, 1):
                        fit, val = train.iloc[fitting], train.iloc[validation]
                        inner_fold_ids[validation] = inner_fold
                        inner_membership.extend({'experiment': experiment, 'outer_fold': fold,
                                                 'inner_fold': inner_fold, 'RID': rid} for rid in val.index)
                        anchor = make_pipeline('logistic_regression', features, original['config'])
                        anchor.fit(fit[features], fit.label)
                        preprocess = anchor.named_steps['preprocessing']
                        x, vx = preprocess.transform(fit[features]), preprocess.transform(val[features])
                        for c in cfg['lr_C_candidates']:
                            reference = make_pipeline('logistic_regression', features, original['config'])
                            reference.set_params(model__C=c)
                            reference.fit(fit[features], fit.label)
                            lr_predictions[c][validation] = reference.predict_proba(val[features])[:, 1]
                        for candidate in cfg['candidates']:
                            print(f'  Inner fold {inner_fold}: {candidate["name"]}', flush=True)
                            network, best_epoch, history = fit_network(x, fit.label, candidate,
                                anchor.named_steps['model'], cfg, cfg['seed'], device,
                                cfg['training']['max_epochs'], validation=(vx, val.label))
                            candidate_predictions[candidate['name']][validation] = predict(network, vx, device)
                            epochs_by_candidate[candidate['name']].append(best_epoch)
                            learning.extend({'experiment': experiment, 'outer_fold': fold, 'inner_fold': inner_fold,
                                             'candidate': candidate['name'], 'stage': 'selection',
                                             'seed': cfg['seed'], **row} for row in history)
                            del network
                    losses = [log_loss(train.label, candidate_predictions[c['name']], labels=[0, 1]) for c in cfg['candidates']]
                    winner = cfg['candidates'][int(np.argmin(losses))]
                    epochs = int(np.median(epochs_by_candidate[winner['name']]))
                    lr_losses = [log_loss(train.label, lr_predictions[c], labels=[0, 1]) for c in cfg['lr_C_candidates']]
                    best_c = cfg['lr_C_candidates'][int(np.argmin(lr_losses))]
                    for candidate, loss in zip(cfg['candidates'], losses):
                        search_rows.append({'experiment': experiment, 'outer_fold': fold, 'family': 'hybrid',
                                            'candidate': candidate['name'], 'inner_log_loss': float(loss),
                                            'selected': candidate['name'] == winner['name'],
                                            'inner_best_epochs': json.dumps(epochs_by_candidate[candidate['name']])})
                    for c, loss in zip(cfg['lr_C_candidates'], lr_losses):
                        search_rows.append({'experiment': experiment, 'outer_fold': fold, 'family': 'logistic',
                                            'candidate': str(c), 'inner_log_loss': float(loss), 'selected': c == best_c,
                                            'inner_best_epochs': ''})
                    inner_audit = pd.DataFrame(candidate_predictions, index=train.index)
                    for c, p in lr_predictions.items():
                        inner_audit[f'logistic_C{c}'] = p
                    inner_audit['label'], inner_audit['inner_fold'] = train.label, inner_fold_ids
                    inner_audit.to_csv(out / f'{experiment}_fold{fold}_selection_predictions.csv', index_label='RID')
                    anchor = make_pipeline('logistic_regression', features, original['config'])
                    anchor.fit(train[features], train.label)
                    tuned = make_pipeline('logistic_regression', features, original['config'])
                    tuned.set_params(model__C=best_c)
                    tuned.fit(train[features], train.label)
                    for name, reference in [('logistic_regression', anchor), ('tuned_logistic_regression', tuned)]:
                        append(experiment, name, fold, held, reference.predict_proba(held[features])[:, 1])
                        joblib.dump(reference, out / 'models' / f'{experiment}_fold{fold}_{name}.joblib')
                    preprocess = anchor.named_steps['preprocessing']
                    x, hx = preprocess.transform(train[features]), preprocess.transform(held[features])
                    seed_probabilities = []
                    for seed in cfg['seeds']:
                        network, _, history = fit_network(x, train.label, winner, anchor.named_steps['model'],
                                                          cfg, seed, device, epochs)
                        probability = predict(network, hx, device)
                        append(experiment, f'hybrid_seed{seed}', fold, held, probability)
                        seed_probabilities.append(probability)
                        parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
                        selection_rows.append({'experiment': experiment, 'outer_fold': fold, 'seed': seed,
                                               'candidate': winner['name'], 'refit_epochs': epochs,
                                               'anchor_only': epochs == 0, 'tuned_lr_C': best_c,
                                               'trainable_parameters': parameters,
                                               'attention_gate': float(torch.sigmoid(network.gates[0]).detach().cpu()),
                                               'residual_gate': float(torch.sigmoid(network.gates[1]).detach().cpu())})
                        learning.extend({'experiment': experiment, 'outer_fold': fold, 'inner_fold': 0,
                                         'candidate': winner['name'], 'stage': 'refit', 'seed': seed, **row} for row in history)
                        stem = f'{experiment}_fold{fold}_seed{seed}'
                        torch.save({'state_dict': {key: value.detach().cpu() for key, value in network.state_dict().items()},
                                    'candidate': winner, 'input_columns': int(x.shape[1]), 'features': features,
                                    'seed': seed, 'epochs': epochs}, out / 'models' / f'{stem}.pt')
                        joblib.dump(preprocess, out / 'models' / f'{stem}_preprocessing.joblib')
                        del network
                    append(experiment, 'hybrid_seed_mean', fold, held, np.mean(seed_probabilities, axis=0))
        predictions = pd.DataFrame(predictions)
        if predictions.duplicated(['experiment', 'model', 'RID']).any():
            raise ValueError('Duplicate held-out participant predictions')
        predictions.to_csv(out / 'held_out_predictions.csv', index=False)
        pd.DataFrame(inner_membership).to_csv(out / 'inner_membership.csv', index=False)
        pd.DataFrame(search_rows).to_csv(report / 'search_results.csv', index=False)
        pd.DataFrame(selection_rows).to_csv(report / 'selected_models.csv', index=False)
        pd.DataFrame(learning).to_csv(report / 'learning_curves.csv', index=False)
        results = generate_reports(predictions, cfg, report, 'hybrid')
        paired = []
        for experiment in cfg['experiments']:
            reference = predictions[(predictions.experiment == experiment) &
                                    (predictions.model == 'tuned_logistic_regression')].set_index('RID')
            for name in [f'hybrid_seed{cfg["seeds"][0]}', 'hybrid_seed_mean']:
                group = predictions[(predictions.experiment == experiment) & (predictions.model == name)].set_index('RID')
                aligned = reference.reindex(group.index)
                if not aligned.label.equals(group.label) or not aligned.fold.equals(group.fold):
                    raise ValueError('Tuned-baseline comparison mismatch')
                paired.extend({'experiment': experiment, 'model': name, 'reference': 'tuned_logistic_regression', **row}
                              for row in paired_intervals(group.label, group.probability, aligned.probability, cfg))
        pd.DataFrame(paired).to_csv(report / 'paired_vs_tuned_logistic.csv', index=False)
        target = results[['experiment', 'model', 'participants', 'accuracy', 'sensitivity', 'specificity']].copy()
        target['target_accuracy'] = cfg['target_accuracy']
        target['reached_target'] = target.accuracy >= cfg['target_accuracy']
        target.to_csv(report / 'target_summary.csv', index=False)
        write_summary(results, cfg, report)
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
