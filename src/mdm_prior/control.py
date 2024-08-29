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

from src.mdm_prior.diffusion.gaussian_diffusion_inpaint import GaussianDiffusionInpainting
from src.mdm_prior.diffusion.respace import SpacedDiffusion

from src.mdm_prior.utils.model_util import load_model_blending_and_diffusion
from src.mdm_prior.utils.parser_util import edit_inpainting_args

from src.mdm_prior.model.model_blending import ModelBlender
from src.mdm_prior.model.cfg_sampler import wrap_model

from src.mdm_prior.data_loaders.get_data import get_dataset_loader
from src.mdm_prior.data_loaders.humanml_utils import get_inpainting_mask
from src.mdm_prior.data_loaders.humanml.scripts.motion_process import recover_from_ric


def main(args_list):
    # args_list = edit_inpainting_args()
    args = args_list[0]
    fix_seed(args.seed)
    
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name = args.dataset,
                      dataset_path = args.data_dir,
                        batch_size = args.batch_size,
                        num_frames = max_frames,
                              size = args.num_samples,
                         load_mode = 'train',   # train mode to get text and motion.
                             split = 'test',)
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    DiffusionClass = GaussianDiffusionInpainting if args_list[0].filter_noise else \
                       SpacedDiffusion
    model, diffusion = load_model_blending_and_diffusion(args_list, data, dist_util.dev(), 
                                                         DiffusionClass=DiffusionClass)

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())
    if args.text_condition != '':
        texts = [args.text_condition] * args.num_samples
        model_kwargs['y']['text'] = texts

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    model_kwargs['y']['inpainted_motion'] = input_motions
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())

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
            init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        n_joints = None
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

    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_motions = np.concatenate(all_motions, axis=0)[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]

    return all_motions, all_text, all_lengths, \
        input_motions, data, model_kwargs, n_joints, fps
            


if __name__ == "__main__":
    main()