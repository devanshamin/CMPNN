import warnings
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import json
from pathlib import Path

from rdkit import RDLogger  
import numpy as np

from cmpnn.train.run_training import run_training
from cmpnn.data.utils import get_task_names
from cmpnn.utils import makedirs, create_logger
from cmpnn.args import parse_train_args

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(1, args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_training(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score


def main():

    args = parse_train_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    Path(args.save_dir, "args.json").write_text(json.dumps({k: v for k, v in vars(args).items() if v}, indent=2))
    mean_auc_score, std_auc_score = cross_validate(args, logger)
    print(f'Results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}')


if __name__ == '__main__':
    
    main()