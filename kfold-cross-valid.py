# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import time
import copy
import os
import os.path as osp
# import numpy as np
# import torch
from mmengine.config import Config, DictAction, ConfigDict
from mmengine.fileio import dump, load
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
# from mmengine.hooks import Hook
# from mmengine.runner import Runner
from mmengine.logging import print_log
from seg.utils import register_all_modules
from seg.utils.train_single_fold import train_single_fold
import threading
import wandb
import gc
EXP_INFO_FILE = 'kfold_exp.json'

prog_description = """K-Fold cross-validation.

To start a 5-fold cross-validation experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5

To resume a 5-fold cross-validation from an interrupted experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5 --resume
"""  # noqa: E501


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=prog_description)
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-splits',
        default=5,
        type=int,
        help='The number of all folds.')
    parser.add_argument(
        '--fold',
        type=int,
        help='The fold used to do validation. '
        'If specify, only do an experiment of the specified fold.')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default='0', help='random seed for split dataset')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume the previous experiment.')
    parser.add_argument(
        '--experiment-name',
        help='Name of resumed experiment.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # set dataloader args
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        default_dataloader_cfg.persistent_workers = False

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = copy.deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False
        if args.no_persistent_workers:
            cfg[field]['persistent_workers'] = False

    set_default_dataloader_cfg(cfg, 'train_dataloader')
    set_default_dataloader_cfg(cfg, 'val_dataloader')
    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg

if __name__ == '__main__':
    # main()
    args = parse_args()

    register_all_modules(False)
    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # set the unify random seed
    cfg.kfold_split_seed = args.seed

    # resume from the previous experiment
    if args.resume:
        assert isinstance(args.experiment_name, str)
        experiment_name = args.experiment_name
        cfg.work_dir = osp.join(cfg.work_dir,
                                experiment_name)
        experiment_info = load(osp.join(cfg.work_dir, EXP_INFO_FILE))
        cfg.kfold_split_seed = int(experiment_info['exp_info']['kfold_split_seed'])
        folds_info = experiment_info['fold_info']
        last_fold = list(folds_info.keys())[-1].split('fold')[-1]
        resume_fold = int(last_fold) + 1
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        experiment_name = f'{args.num_splits}fold_cross_valid_{timestamp}'
        cfg.work_dir = osp.join(cfg.work_dir,
                                experiment_name)
        resume_fold = 0

    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = range(resume_fold, args.num_splits)

    for fold in folds:
        os.environ['WANDB_MODE'] = 'offline'
        # cfg.train_cfg.max_iters = 50
        # cfg.train_cfg.val_interval = 50
        cfg_ = copy.deepcopy(cfg)
        train_single_fold(cfg_, args.num_splits, fold, experiment_name, args.resume)
        # t1.is_alive()
        # with train_single_fold(cfg_, args.num_splits, fold, experiment_name, args.resume):
        #     pass
        # gc.collect()
        if wandb.run is not None:
            # path = osp.split(wandb.run.dir)[0]
            wandb.join()
            # a = subprocess.run(("wandb", "sync", path))


