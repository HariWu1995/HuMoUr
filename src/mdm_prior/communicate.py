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
import src.utils.rotation_conversions as geometry

from src.mdm_prior.utils.model_util import create_model_and_diffusion, load_model
from src.mdm_prior.utils.parser_util import generate_multi_args

from src.mdm_prior.model.comMDM import ComMDM
from src.mdm_prior.model.cfg_sampler import ClassifierFreeSampleModel

from src.mdm_prior.data_loaders.tensors import collate
from src.mdm_prior.data_loaders.get_data import get_dataset_loader
from src.mdm_prior.data_loaders.humanml.scripts.motion_process import recover_from_ric


def main(args):
    
    # print(f"generating samples")
    # args = generate_multi_args()
    args.guidance_param = 1.  # Hard coded - higher values will work but will limit diversity.
    fix_seed(args.seed)
    
    sample1 = None
    max_frames = 120
    n_frames = 120
    fps = 20
    is_using_data = not any([args.input_text, args.text_prompt])
    dist_util.setup_dist(args.device)

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1

    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    if not args.sample_gt:
        model, diffusion = load_model(args, data, dist_util.dev(), ModelClass=ComMDM)
    else:
        model, diffusion = create_model_and_diffusion(args, data, ModelClass=ComMDM)

    if is_using_data:
        iterator = iter(data)
        gt_motion, model_kwargs = next(iterator)
        n_frames = int(max(model_kwargs['y']['lengths']))

    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []
    all_captions = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

        if not args.sample_gt:
            sample_fn = diffusion.p_sample_loop
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames+1),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                predict_two_person=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample, sample1 = sample
        else:
            sample = gt_motion.cpu()
            sample1 = model_kwargs['y']['other_motion'].cpu()

        canon0, sample = torch.split(sample, [1, sample.shape[-1] - 1], dim=-1)
        canon1, sample1 = torch.split(sample1, [1, sample1.shape[-1] - 1], dim=-1)

        canon0 = data.dataset.t2m_dataset.rebuilt_canon(canon0[:, :4, 0, 0])
        canon1 = data.dataset.t2m_dataset.rebuilt_canon(canon1[:, :4, 0, 0])

        diff_trans = canon1[:, -3:] - canon0[:, -3:]

        _rot0 = geometry.rotation_6d_to_matrix(canon0[:, :6])
        _rot1 = geometry.rotation_6d_to_matrix(canon1[:, :6])

        diff_rot = torch.matmul(_rot0, _rot1.permute(0, 2, 1)).float().cpu()

        # Recover XYZ *positions* from HumanML3D vector representation
        n_joints = None
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            if sample1 is not None:
                sample1 = data.dataset.t2m_dataset.inv_transform(sample1.cpu().permute(0, 2, 3, 1)).float()
                sample1 = recover_from_ric(sample1, n_joints)
                sample1 = torch.matmul(diff_rot.view(-1, 1, 1, 1, 3, 3), sample1.unsqueeze(-1)).squeeze(-1)
                sample1 = sample1.view(-1, *sample1.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
                sample1 += diff_trans.view(-1, 1, 3, 1).cpu().numpy()

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
            data, model_kwargs, n_joints, fps


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name = args.multi_dataset,
                      dataset_path = args.data_dir,
                        batch_size = args.batch_size,
                        num_frames = max_frames,
                             split = 'validation', # test
                         load_mode = 'text',)  # for GT vis
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()