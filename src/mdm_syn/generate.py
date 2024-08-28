# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large numpy array. 
This can be used to produce samples for FID evaluation.
"""
import os
import shutil
import copy

import numpy as np
import torch

from src.utils import dist_util
from src.utils.seeding import fix_seed

from src.mdm_syn.utils.parser_util import generate_args
from src.mdm_syn.utils.model_util import create_model_and_diffusion, load_model

from src.mdm_syn.data_loaders.tensors import collate
from src.mdm_syn.data_loaders.get_data import get_dataset_loader
from src.mdm_syn.data_loaders.mixamo.motion import MotionData
from src.mdm_syn.data_loaders.humanml.scripts.motion_process import recover_from_ric
from src.mdm_syn.data_loaders.humanml.utils.plot_script import plot_3d_motion
import src.mdm_syn.data_loaders.humanml.utils.paramUtil as paramUtil

from src.mdm_syn.motion import BVH
from src.mdm_syn.motion.transforms import repr6d2quat
from src.mdm_syn.motion.Quaternions import Quaternions
from src.mdm_syn.motion.Animation import Animation, positions_global as anim_pos
from src.mdm_syn.motion.AnimationStructure import get_kinematic_chain


def main(args):
    # args = generate_args()
    fix_seed(args.seed)
    dist_util.setup_dist(args.device)
    
    motion_data = None
    num_joints = None
    repr = 'repr6d' if args.repr == '6d' else 'quat'

    if args.dataset == 'mixamo':
        motion_data = MotionData(args.sin_path, padding=True, use_velo=True,
                                 repr=repr, contact=True, keep_y_pos=True,
                                 joint_reduction=True)
        fps = int(round(1 / motion_data.bvh_file.frametime))
        n_frames = motion_data.bvh_file.anim.shape[0]
        skeleton = get_kinematic_chain(motion_data.bvh_file.skeleton.parent)
        num_joints = motion_data.raw_motion.shape[1]

    elif args.dataset == 'bvh_general':
        sin_anim, joint_names, frametime = BVH.load(args.sin_path)
        fps = int(round(1 / frametime))
        skeleton = get_kinematic_chain(sin_anim.parents)
        n_frames = sin_anim.shape[0]
        num_joints = sin_anim.shape[1]

    else:
        assert args.dataset == 'humanml'
        fps = 20
        skeleton = paramUtil.t2m_kinematic_chain

    max_frames = 196 if args.dataset == 'humanml' else 60
    if args.motion_length is not None:
        n_frames = int(args.motion_length*fps)
    elif not args.dataset in ['mixamo', 'bvh_general']:
        n_frames = max_frames

    # TODO: fix this hack. not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    is_using_data = False

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger than default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    total_num_samples = args.num_samples

    if args.dataset in ['humanml']:
        print('Loading dataset...')
        data = load_dataset(args, max_frames, n_frames)
    else:
        data = None

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, motion_data, num_joints)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    diffusion.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  # disable random masking
    model.requires_grad_(False)

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 
                        'tokens': None, 
                        'lengths': n_frames}] * args.num_samples
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []

    print(f'### Sampling')

    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
        model,
        (args.batch_size, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    sample = postprocess(sample, model, args, prefix_save='sample_')

    if args.unconstrained:
        all_text += ['generated'] * args.num_samples
    else:
        text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
        all_text += model_kwargs['y'][text_key]

    if isinstance(sample, torch.Tensor):
        sample = sample.cpu().numpy()
    all_motions.append(sample)
    all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

    print(f"created {len(all_motions) * args.batch_size} samples")

    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text    =    all_text[:total_num_samples]

    return all_motions, all_text, all_lengths


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


def postprocess(motions, model, args, prefix_save: str = 'prefix_'):

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        motions = data.dataset.t2m_dataset.inv_transform(motions.cpu().permute(0, 2, 3, 1)).float()
        motions = recover_from_ric(motions, n_joints)
        motions = motions.view(-1, *motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    
    # Recover XYZ *positions* from zoo vector representation
    elif model.data_rep in ['mixamo_vec', 'bvh_general_vec']:
        motions = motions.cpu().numpy()
        motions = motions.transpose(0, 3, 1, 2)  # n_samples x n_joints x n_features x n_frames  ==>   n_samples x n_frames x n_joints x n_features
        
        if args.dataset == 'mixamo':
            xyz_samples = np.zeros((args.num_samples, motions.shape[1], 24, 3))  # shape it to match the output of anim_pos
        else:
            joint_features_length = 7 if args.repr == 'quat' else 9
            assert model.njoints % joint_features_length == 0
            xyz_samples = np.zeros((args.num_samples, n_frames, int(model.njoints / joint_features_length), 3))  # shape it to match the output of anim_pos

        for i, one_sample in enumerate(motions):
            bvh_path = os.path.join(args.out_path, f'{prefix_save}{0:02d}.bvh')

            if args.dataset == 'mixamo':
                motion_data.write(bvh_path, torch.tensor(one_sample.transpose((2, 1, 0))))
                generated_motion = MotionData(bvh_path, padding=True, use_velo=True, repr='repr6d', 
                                                       contact=True, keep_y_pos=True, joint_reduction=True)
                anim = Animation(rotations=Quaternions(
                                           generated_motion.bvh_file.get_rotation().numpy()),
                                 positions=generated_motion.bvh_file.anim.positions,
                                   orients=generated_motion.bvh_file.anim.orients,
                                   offsets=generated_motion.bvh_file.skeleton.offsets,
                                   parents=generated_motion.bvh_file.skeleton.parent)
            else:
                if args.repr == '6d':
                    one_sample = one_sample.reshape(n_frames, -1, joint_features_length)
                    quats = repr6d2quat(torch.tensor(one_sample[:, :, 3:])).numpy()
                else:
                    quats = one_sample[:, :, 3:]

                anim = Animation(rotations=Quaternions(quats), 
                                 positions=one_sample[:, :, :3],
                                   orients=sin_anim.orients, 
                                   offsets=sin_anim.offsets, 
                                   parents=sin_anim.parents)

            xyz_samples[i] = anim_pos(anim)  # n_frames x n_joints x 3  =>
        
        # n_samples x n_frames x n_joints x 3  =>  n_samples x n_joints x 3 x n_frames
        motions = xyz_samples.transpose(0, 2, 3, 1)
    
    return motions


if __name__ == "__main__":
    main()
