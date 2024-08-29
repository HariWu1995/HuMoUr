# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large numpy array. 
This can be used to produce samples for FID evaluation.
"""
import os
import copy
import shutil
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import dist_util
from src.utils.seeding import fix_seed

from src.mdm_syn.utils.parser_util import edit_args
from src.mdm_syn.utils.model_util import create_model_and_diffusion, load_model

import src.mdm_syn.data_loaders.humanml.utils.paramUtil as paramUtil
from src.mdm_syn.data_loaders.humanml import humanml_utils
from src.mdm_syn.data_loaders.humanml.scripts.motion_process import recover_from_ric
from src.mdm_syn.data_loaders.mixamo import mixamo_utils
from src.mdm_syn.data_loaders.mixamo.motion import MotionData
from src.mdm_syn.data_loaders.data_util import load_sin_motion

from src.mdm_syn.motion import BVH
from src.mdm_syn.motion.transforms import repr6d2quat
from src.mdm_syn.motion.Quaternions import Quaternions
from src.mdm_syn.motion.Animation import Animation, positions_global as anim_pos
from src.mdm_syn.motion.AnimationStructure import get_kinematic_chain


def main(args):
    # args = edit_args()
    fix_seed(args.seed)
    dist_util.setup_dist(args.device)

    num_joints = None
    max_frames = 196 if args.dataset == 'humanml' else 60
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
        skeleton = paramUtil.t2m_kinematic_chain
        fps = 20

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger than default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)

    # Sampling a single batch from the testset, with exactly args.num_samples
    args.batch_size = args.num_samples  
    
    if args.dataset in ['humanml']:
        print('Loading dataset...')
        n_frames = max_frames
        data = load_dataset(args, max_frames, n_frames)
    else:
        data = None

    total_num_samples = args.num_samples
    if args.ref_motion != "":
        args.sin_path = args.ref_motion

    motion, motion_data = load_sin_motion(args)  # shape (joints, frames)
    if args.dataset == 'bvh_general':
        motion = motion.reshape(1, -1, n_frames)

    if args.num_frames != -1:
        motion = motion[..., :args.num_frames]
        if motion.shape[-1] < args.num_frames:
            last_frame = copy.deepcopy(motion[:, :, -2:-1])
            motion_suffix = last_frame.repeat((1, 1, int(args.num_frames-motion.shape[-1])))
            motion = torch.cat((motion, motion_suffix), 2)
            n_frames = args.num_frames

    input_motions = motion.repeat((args.num_samples, 1, 1, 1))
    input_motions = input_motions.transpose(1,2)
    motion = input_motions[0]

    print("Creating model and diffusion...")
    args.unconstrained = True
    model, diffusion = create_model_and_diffusion(args, data, num_joints)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    model.eval()  # disable random masking

    num_rows = args.num_samples
    if args.edit_mode == 'expansion':
        first_frame = copy.deepcopy(motion[:, :, 0:1])
        last_frame = copy.deepcopy(motion[:, :, -2:-1])

        first_frame[-4, ...] = 0
        first_frame[-6, ...] = 0
        last_frame[-4, ...] = 0
        last_frame[-6, ...] = 0

        motion_prefix = first_frame.repeat((1, 1, int(motion.shape[-1] * args.prefix_length)))
        motion_suffix = last_frame.repeat((1, 1, int(motion.shape[-1] * args.suffix_length)))
        outpaint_upto_idx = motion_prefix.shape[-1]
        outpaint_from_idx = motion_prefix.shape[-1] + motion.shape[-1]
        motion = torch.cat((motion_prefix, motion, motion_suffix), 2)

        n_frames = motion.shape[-1]

    input_motions = motion.repeat((num_rows, 1, 1, 1))
    model_kwargs = {}
    model_kwargs['y'] = {}
    model_kwargs['y']['lengths'] = torch.tensor(motion.shape[2]).repeat((num_rows))

    resizers = range_t = None

    # add inpainting mask according to args
    max_frames = input_motions.shape[-1]
    assert max_frames == input_motions.shape[-1]
    
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions # => samples, joints, 1, frames
    
    if args.edit_mode == 'in_between':
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.float,
                                                               device=input_motions.device)  # True means use gt motion
        for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
            start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
            gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
            
            # do inpainting in those frames
            model_kwargs['y']['inpainting_mask'][i, :, :, start_idx: end_idx] = False
            
            mask_slope = 10
            for f in range(mask_slope):

                if start_idx-f < 0:
                    continue
                model_kwargs['y']['inpainting_mask'][i, :, :, start_idx-f] = f / mask_slope

                if end_idx+f >= length:
                    continue
                model_kwargs['y']['inpainting_mask'][i, :, :, end_idx+f] = f / mask_slope

    elif args.edit_mode == 'upper_body':
        assert args.dataset in ['humanml', 'mixamo'], \
            "Editing upper body is only supported for humanml or mixamo data"

        if args.dataset == 'humanml':
            model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK, dtype=torch.bool,
                                                                device=input_motions.device)  # True is lower body data
        elif args.dataset == 'mixamo':
            model_kwargs['y']['inpainting_mask'] = torch.tensor(mixamo_utils.mixamo_lower_body_mask(repr), dtype=torch.bool,
                                                                device=input_motions.device)  # True is lower body data

        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].repeat(input_motions.shape[0], 1, 
                                                                                           input_motions.shape[2], 
                                                                                           input_motions.shape[3])
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].float()
    
    elif args.edit_mode == 'lower_body':
        assert args.dataset in ['humanml', 'mixamo'], \
            "Editing lower body is only supported for humanml or mixamo data"

        if args.dataset == 'humanml':
            model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_UPPER_BODY_MASK, dtype=torch.bool,
                                                                device=input_motions.device)  # True is lower body data
        elif args.dataset == 'mixamo':
            model_kwargs['y']['inpainting_mask'] = torch.tensor(mixamo_utils.mixamo_upper_body_mask(repr), dtype=torch.bool,
                                                                device=input_motions.device)  # True is lower body data
        
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].repeat(input_motions.shape[0], 1, 
                                                                                           input_motions.shape[2], 
                                                                                           input_motions.shape[3])
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].float()

    elif args.edit_mode == 'expansion':
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.float,
                                                               device=input_motions.device)  # True means use gt motion
        
        for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
            gt_frames_per_sample[i] = list(range(outpaint_upto_idx, outpaint_from_idx))
            
            model_kwargs['y']['inpainting_mask'][i, :, :, : outpaint_upto_idx] = False  # do outpainting in those frames
            model_kwargs['y']['inpainting_mask'][i, :, :, outpaint_from_idx :] = False  # do outpainting in those frames
            
            mask_slope = 6
            for f in range(mask_slope):

                if outpaint_upto_idx + f >= length:
                    continue
                model_kwargs['y']['inpainting_mask'][i, :, :, outpaint_upto_idx + f] = f / mask_slope
                
                if outpaint_from_idx - f < 0:
                    continue
                model_kwargs['y']['inpainting_mask'][i, :, :, outpaint_from_idx - f] = f / mask_slope
    
    elif args.edit_mode == 'harmonization':
        print("Creating ressizers ...")
        if input_motions.shape[-1] % 2 != 0:
            input_motions = input_motions[..., :-1]
            n_frames -= 1

        scale = 2
        range_t = 20
        down = partial(F.interpolate, mode='bilinear', align_corners=False, scale_factor=(1, 1/scale))
        up = partial(F.interpolate, mode='bilinear', align_corners=False, scale_factor=(1, scale))
        resizers = (down, up)
        max_frames = input_motions.shape[-1]
        model_kwargs["y"]["ref_img"] = input_motions

    all_motions = []
    all_lengths = []
    all_text = []

    print(f'### Start sampling')

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
        resizers=resizers,
        range_t=range_t
    )

    sample = postprocess(sample, model, data, args, prefix_save='sample_')

    if args.unconstrained:
        all_text.extend([args.edit_mode] * args.num_samples)
    else:
        all_text += model_kwargs['y']['text']

    if isinstance(sample, torch.Tensor):
        sample = sample.cpu().numpy()
    all_motions.append(sample)
    all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

    print(f"created {len(all_motions) * args.batch_size} samples")

    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text    =    all_text[:total_num_samples]

    input_motions = postprocess(input_motions, model, data, args, prefix_save='input_')

    return all_motions, all_text, all_lengths, \
         input_motions, skeleton, data, gt_frames_per_sample, \
        (22 if motions.shape[1] == 263 else 21) 
            if model.data_rep == 'hml_vec' else num_joints, fps


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


def postprocess(motions, model, data, args, prefix_save: str = 'prefix_'):

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        n_joints = 22 if motions.shape[1] == 263 else 21
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
