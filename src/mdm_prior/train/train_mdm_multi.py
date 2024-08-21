# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json

import torch

from src.utils import dist_util
from src.utils.seeding import fix_seed

from src.mdm_prior.model.comMDM import ComMDM

from src.mdm_prior.utils.model_util import create_model_and_diffusion, load_model_wo_clip, load_pretrained_mdm, load_split_mdm
from src.mdm_prior.utils.parser_util import train_multi_args

from src.mdm_prior.train.training_loop import TrainLoop
from src.mdm_prior.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

from src.mdm_prior.data_loaders.get_data import get_dataset_loader


def main():
    args = train_multi_args()
    fix_seed(args.seed)

    if args.multi_train_mode == 'prefix':
        args.cond_mask_prob = 1.  # Hard-coded! We learn unconditioned in this setting!

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')

    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))

    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.multi_dataset, batch_size=args.batch_size, num_frames=args.num_frames, split=args.multi_train_splits, load_mode=args.multi_train_mode)

    print("Creating model and diffusion...")
    ModelClass = ComMDM
    model, diffusion = create_model_and_diffusion(args, data, ModelClass)

    print(f"Loading checkpoints from [{args.pretrained_path}]...")
    state_dict = torch.load(args.pretrained_path, map_location='cpu')
    if args.multi_backbone_split == 0:
        load_pretrained_mdm(model, state_dict)
    else:
        load_split_mdm(model, state_dict, args.multi_backbone_split)

    model.to(dist_util.dev())

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print('Trainable params: %.2fM' % (sum(p.numel() for p in model.trainable_parameters()) / 1000000.0))
    print('Multi-Person params: %.2fM' % (sum(p.numel() for p in model.multi_parameters()) / 1000000.0))
    assert sum(p.numel() for p in model.multi_parameters()) == sum(p.numel() for p in model.trainable_parameters())

    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
