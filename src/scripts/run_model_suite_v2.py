#!/usr/bin/env python3
"""Run an existing model family on saved v2 participant folds, without notebooks."""
import argparse
import json

from train_models import local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', default='config/suite-primary')
    parser.add_argument('--method', required=True,
                        choices=['voting', 'stacking', 'mlp', 'resnet', 'tabpfn', 'hybrid', 'lr_stacking'])
    args = parser.parse_args()
    family = ('ensembles' if args.method in {'voting', 'stacking'} else
              'deep_learning' if args.method in {'mlp', 'resnet', 'tabpfn'} else
              'hybrid_attention' if args.method == 'hybrid' else 'lr_stacking')
    cfg = json.loads((local_path(args.config_dir) / (family + '.json')).read_text())
    if cfg.get('input_format') != 'v2':
        raise ValueError('This runner requires v2 suite configurations')
    # Lazy imports keep the tree/LR commands independent of PyTorch and TabPFN.
    if args.method in {'voting', 'stacking'}:
        from ensemble_models import run_experiment
        run_experiment(cfg, args.method)
    elif args.method in {'mlp', 'resnet', 'tabpfn'}:
        from deep_models import run_deep_experiment
        run_deep_experiment(cfg, args.method)
    elif args.method == 'hybrid':
        from hybrid_models import run_hybrid
        run_hybrid(cfg)
    else:
        from lr_stacking import run_lr_stacking
        run_lr_stacking(cfg)


if __name__ == '__main__':
    main()
