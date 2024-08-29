# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import shutil

import numpy as np
import torch

from src.utils import dist_util
from src.utils.seeding import fix_seed

from src.mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from src.mdm.utils.parser_util import edit_args

from src.mdm.model.cfg_sampler import ClassifierFreeSampleModel

from src.mdm.data_loaders import humanml_utils
from src.mdm.data_loaders.get_data import get_dataset_loader
from src.mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric


def main(args):
    # args = edit_args()
    fix_seed(args.seed)
    
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size ({args.batch_size}) or reduce num_samples ({args.num_samples})'
    
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name = args.dataset,
                        batch_size = args.batch_size,
                      dataset_path = args.data_dir,
                        num_frames = max_frames,
                             split = 'test',
                          hml_mode = 'train')  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())
    texts = [args.text_condition] * args.num_samples
    model_kwargs['y']['text'] = texts

    if args.text_condition == '':
        args.guidance_param = 0.  # Force unconditioned generation

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1], \
        f"`max_frames` ({max_frames}) MUST be equal to {input_motions.shape[-1]}"
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions
    
    if args.edit_mode == 'in_between':
        # True means use gt motion
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, 
                                                               dtype=torch.bool,
                                                               device=input_motions.device)  
        for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
            start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
            gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))

            # do inpainting in those frames
            model_kwargs['y']['inpainting_mask'][i, :, :, start_idx:end_idx] = False  

    elif args.edit_mode == 'upper_body':
        # True is lower body data
        model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK, 
                                                            dtype=torch.bool,
                                                            device=input_motions.device)
        model_kwargs['y']['inpainting_mask'] = \
        model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)\
                                            .repeat(input_motions.shape[0], 1, 
                                                    input_motions.shape[2], 
                                                    input_motions.shape[3])
    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        length    = model_kwargs['y']['lengths']
        all_lengths.append(length.cpu().numpy())
        all_motions.append(sample.cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    return all_motions, all_text, all_lengths, \
            data, model, model_kwargs, gt_frames_per_sample, n_joints, fps

