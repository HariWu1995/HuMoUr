# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import shutil
import numpy as np
import pandas as pd
import torch

from src.utils import dist_util
from src.utils.seeding import fix_seed

from src.mdm_prior.utils.model_util import load_model
from src.mdm_prior.utils.parser_util import generate_args

from src.mdm_prior.model.DoubleTake_MDM import doubleTake_MDM
from src.mdm_prior.model.cfg_sampler import ClassifierFreeSampleModel

from src.mdm_prior.data_loaders.get_data import get_dataset_loader
from src.mdm_prior.data_loaders.humanml.scripts.motion_process import recover_from_ric

from src.mdm_prior.utils.sampling_utils import unfold_sample_arb_len, double_take_arb_len


def main(args):

    # print(f"generating samples")
    # args = generate_args()
    fix_seed(args.seed)

    fps = 30 if args.dataset == 'babel' else 20
    n_frames = 150

    dist_util.setup_dist(args.device)

    is_using_data = not (args.input_text or args.text_prompt)
    is_csv, is_txt = False, False
    assert (args.double_take), "Please set `double_take` be TRUE"

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        is_txt = True
        texts = args.text_prompt.split('.')
        args.num_samples = len(texts)

    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        if is_txt:
            with open(args.input_text, 'r') as fr:
                texts = fr.readlines()
            texts = [s.replace('\n', '') for s in texts]
            args.num_samples = len(texts)

        elif is_csv:
            df = pd.read_csv(args.input_text)
            args.num_samples = len(list(df['text']))

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset ...')
    data = load_dataset(args, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = load_model(args, data, dist_util.dev(), ModelClass=doubleTake_MDM)

    if is_using_data:
        iterator = iter(data)
        gt_motion, model_kwargs = next(iterator)

    elif is_csv:
        model_kwargs = {'y': {
            'mask': torch.ones((len(list(df['text'])), 1, 1, 196)), #196 is humanml max frames number
            'lengths': torch.tensor(list(df['length'])),
            'text': list(df['text']),
            'tokens': [''],
            'scale': torch.ones(len(list(df['text'])))*2.5,
        }}

    elif is_txt:
        model_kwargs = {'y': {
            'mask': torch.ones((len(texts), 1, 1, 196)), # 196 is humanml max frames number
            'lengths': torch.tensor([n_frames]*len(texts)),
            'text': texts,
            'tokens': [''],
            'scale': torch.ones(len(texts))*2.5,
        }}

    else:
        raise TypeError("Only text-to-motion is availible!")

    all_motions = []
    all_lengths = []
    all_text = []
    all_captions = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        model_kwargs['y'] = {
                key: val.to(dist_util.dev()) if torch.is_tensor(val) else val 
            for key, val in model_kwargs['y'].items()
        }

        max_arb_len = model_kwargs['y']['lengths'].max()
        min_arb_len = 2 * args.handshake_size + 2*args.blend_len + 10

        for ii, len_s in enumerate(model_kwargs['y']['lengths']):
            if len_s > max_arb_len:
                model_kwargs['y']['lengths'][ii] = max_arb_len
            if len_s < min_arb_len:
                model_kwargs['y']['lengths'][ii] = min_arb_len

        samples_per_rep_list, \
        samples_type = double_take_arb_len(args, diffusion, model, model_kwargs, max_arb_len)

        step_sizes = np.zeros(len(model_kwargs['y']['lengths']), dtype=int)
        for ii, len_i in enumerate(model_kwargs['y']['lengths']):
            if ii == 0:
                step_sizes[ii] = len_i
                continue
            step_sizes[ii] = step_sizes[ii-1] + len_i - args.handshake_size

        final_n_frames = step_sizes[-1]

        for sample_i, samples_type_i in zip(samples_per_rep_list, samples_type):

            sample = unfold_sample_arb_len(sample_i, args.handshake_size, step_sizes, final_n_frames, model_kwargs)

            # Recover XYZ *positions* from HumanML3D vector representation
            n_joints = None
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            if args.dataset == 'babel':

                from data_loaders.amass.transforms import SlimSMPLTransform
                transform = SlimSMPLTransform(batch_size=args.batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)

                all_feature = sample                                        # [bs, nfeats, 1, seq_len]
                all_feature_squeeze = all_feature.squeeze(2)                # [bs, nfeats, seq_len]
                all_feature_permutes = all_feature_squeeze.permute(0, 2, 1) # [bs, seq_len, nfeats]
                
                splitted = torch.split(all_feature_permutes, all_feature.shape[0]) #[list of [seq_len,nfeats]]
                sample_list = []
                for seq in splitted[0]:
                    all_features = seq
                    Datastruct = transform.SlimDatastruct
                    datastruct = Datastruct(features=all_features)
                    sample = datastruct.joints

                    sample_list.append(sample.permute(1, 2, 0).unsqueeze(0))
                sample = torch.cat(sample_list)

            else:
                rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
                if args.dataset == 'babel':
                    rot2xyz_pose_rep = 'rot6d'
                rot2xyz_mask = None

                sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, 
                                       pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                       jointstype='smpl', vertstrans=True, betas=None, beta=0, 
                                       glob_rot=None, get_rotations_back=False)

            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'

            all_text     += model_kwargs['y'][text_key]
            all_captions += model_kwargs['y'][text_key]

            length = model_kwargs['y']['lengths']
            all_lengths.append(length.cpu().numpy())
            all_motions.append(sample.cpu().numpy())

            print(f"created {len(all_motions) * args.batch_size} samples")

    # param update for unfolding visualization
    # out of for rep_i
    old_num_samples = args.num_samples
    args.num_samples = 1
    args.batch_size = 1
    n_frames = final_n_frames

    all_lengths = [n_frames] * args.num_repetitions
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text    =    all_text[:total_num_samples]
    
    return all_motions, all_text, all_lengths, \
            data, model_kwargs, samples_type, n_joints, step_sizes, fps


def load_dataset(args, n_frames):
    if args.dataset == 'babel':
        args.num_frames = (args.min_seq_len, args.max_seq_len)
    else:
        args.num_frames = n_frames

    data = get_dataset_loader(name = args.dataset,
                      dataset_path = args.data_dir,
                        batch_size = args.batch_size,
                        num_frames = args.num_frames,
                          short_db = args.short_db,
                  cropping_sampler = args.cropping_sampler,
                             split = 'val',
                         load_mode = 'text_only',)
                              
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
