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

from src.mdm_prior.utils.parser_util import edit_multi_args
from src.mdm_prior.utils.model_util import load_model

from src.mdm_prior.model.comMDM import ComMDM
from src.mdm_prior.model.cfg_sampler import UnconditionedModel

from src.mdm_prior.data_loaders.get_data import get_dataset_loader
from src.mdm_prior.data_loaders import humanml_utils

from src.mdm_prior.eval.eval_multi import extract_motions


def main(args):

    # print(f"generating samples")
    # args = edit_multi_args()
    fix_seed(args.seed)

    fps = 20
    n_frames = 80
    max_frames = n_frames + 1  # for global root pose
    sample1 = None  # a place holder for two characters, do not delete

    args.num_repetitions = 1            # Hardcoded - prefix completion has limited diversity.
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    args.edit_mode = 'prefix'           # Prefix completion script.

    dist_util.setup_dist(args.device)

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    args.guidance_param = 0
    model, diffusion = load_model(args, data, dist_util.dev(), ModelClass=ComMDM)

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions[..., :n_frames+1]
    other_motions = model_kwargs['y']['other_motion']

    model_kwargs['y']['mask'        ] = model_kwargs['y']['mask'        ][..., :n_frames+1]
    model_kwargs['y']['other_motion'] = model_kwargs['y']['other_motion'][..., :n_frames+1]
    model_kwargs['y']['inpainted_motion_multi'] = [input_motions, 
                                                   other_motions.to(input_motions.device)]
    model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, 
                                                            dtype=torch.bool,
                                                           device=input_motions.device)  # True means use gt motion
    
    gt_frames_per_sample = {}
    for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
        start_idx = int(args.prefix_end * length)
        end_idx = int(args.suffix_start * length)
        gt_frames_per_sample[i] = list(range(0, start_idx))
        # do inpainting in those frames
        model_kwargs['y']['inpainting_mask'][i, :, :, start_idx+1:] = False

    all_motions = []
    all_lengths = []
    all_text = []
    all_captions = []

    gt, gt1 = extract_motions(input_motions.cpu(), 
                              other_motions.cpu(), data)

    def process_to_save(_sample0, _sample1):
        sample_save = np.concatenate((_sample0[None], 
                                      _sample1[None]), axis=0).transpose(1, 0, 4, 2, 3)
        return sample_save.reshape(*sample_save.shape[:3], -1)

    gt_save = process_to_save(gt, gt1)

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val 
                         for key, val in model_kwargs['y'].items()}
        model_kwargs['y']['inpainted_motion_multi'][0] = model_kwargs['y']['inpainted_motion_multi'][0].to(dist_util.dev())
        model_kwargs['y']['inpainted_motion_multi'][1] = model_kwargs['y']['inpainted_motion_multi'][1].to(dist_util.dev())

        other_motions = model_kwargs['y']['other_motion']

        sample_fn = diffusion.p_sample_loop    
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames+1),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=(input_motions, other_motions),
            progress=True,
            predict_two_person=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        sample, sample1 = sample
        sample, sample1 = extract_motions(sample, sample1, data)

        text_key = 'text'
        all_text     += model_kwargs['y'][text_key]
        all_captions += model_kwargs['y'][text_key]

        length = model_kwargs['y']['lengths']
        all_lengths.append(length.cpu().numpy())
        all_motions.append(sample.cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text    =    all_text[:total_num_samples]

    return all_motions, all_text, all_lengths, sample, sample1, \
            data, model_kwargs, gt_save, gt_frames_per_sample, fps


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name = 'pw3d',  # args.multi_dataset,
                      dataset_path = args.data_dir,
                        batch_size = args.batch_size,
                        num_frames = max_frames,
                             split = 'validation',  # args.multi_eval_splits,
                         load_mode = 'prefix')  # for GT vis
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
