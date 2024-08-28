import os
import shutil
import json

from typing import List, Optional, Callable

import numpy as np
import torch

from src.utils import dist_util
from src.utils.seeding import fix_seed

from src.mdm_trans.data_loaders.humanml.scripts.motion_process import recover_from_ric
from src.mdm_trans.data_loaders.tensors import collate, get_cond
from src.mdm_trans.data_loaders.get_data import get_dataset, dataset_loader_from_data

from src.mdm_trans.utils.visualize import plot_3d_motion
from src.mdm_trans.utils.misc import recursive_op1, tensor_to_device


def get_max_frames(dataset, motion_length, n_frames=None):
    if n_frames is not None:
        max_frames = int(n_frames.max().item())
    elif motion_length is not None:
        max_frames = int(round(motion_length * dataset.fps))
    else:
        max_frames = dataset.max_frames
    return max_frames


def get_sample_vars(args: int, data: int, model: int, texts: List[str], 
                    get_out_name: Optional[Callable[[int], str]] = None, 
                    is_using_data: Optional[bool] = False, n_frames = None):

    if args.output_dir == '':
        out_name = get_out_name(args)
        out_dir = os.path.dirname(args.model_path)
        out_path = os.path.join(out_dir, out_name)
    else:
        out_path = args.output_dir

    args.batch_size = args.num_samples

    if is_using_data:
        fix_seed(args.seed)  # re-fix the seed so we get same texts even when running models w/ different architectures
        data_loader = dataset_loader_from_data(data, args.dataset, args.batch_size, num_workers=0)  # use 0 workers to always obtain the same data samples
        iterator = iter(data_loader)
        _, model_kwargs = next(iterator)
        max_frames = model_kwargs['y']['lengths'].max().item()

    else:
        max_frames = get_max_frames(data, args.motion_length, n_frames)
        collate_args = [{'inp': torch.zeros(max_frames), 
                        'tokens': None, 
                        'lengths': max_frames}] * args.num_samples
        collate_args = [dict(arg, text=txt) 
                              for arg, txt in zip(collate_args, texts)]
        if n_frames is not None:
            # relevant when using inversion
            collate_args = [dict(arg, lengths=cur_frames.item()) 
                                     for arg, cur_frames in zip(collate_args, n_frames)]
        _, model_kwargs = collate(collate_args)

    shape = (args.num_samples, model.njoints, model.nfeats, max_frames)
    
    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    model_kwargs = recursive_op1(model_kwargs, tensor_to_device, device=dist_util.dev())

    return out_path, shape, model_kwargs, max_frames


def get_niter(model_path):
    niter = os.path.basename(model_path).replace('model', '').replace('.pt', '')
    return niter


def sample_motions(args, model, shape, model_kwargs, max_frames, init_noise, sample_func):
    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        sample = sample_func(  
            model,
            shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_noise,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            all_text += model_kwargs['y']['text']

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(np.minimum(model_kwargs['y']['lengths'].cpu().numpy(), max_frames))

    all_motions = np.concatenate(all_motions, axis=0)
    total_num_samples = args.num_samples * args.num_repetitions
    
    # if number of requested samples is less than a multiple of the batch size, then trim the results
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text    = all_text[   :total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    return all_motions, all_lengths, all_text


def get_xyz_rep(data, all_motions):

    # Recover XYZ *positions* from HumanML3D vector representation
    all_motions = data.t2m_dataset.inv_transform(all_motions.transpose(0, 2, 3, 1))
    all_motions = recover_from_ric(torch.from_numpy(all_motions), data.n_joints).numpy()
    
    xyz_samples = all_motions.reshape(-1, *all_motions.shape[2:]).transpose(0, 2, 3, 1)
    return xyz_samples

