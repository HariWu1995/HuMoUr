"""
This code is a variation of https://github.com/rubenvillegas/cvpr2018nkn/blob/master/datasets/fbx2bvh.py
"""
import os
import os.path as osp

from glob import glob
from tqdm import tqdm

import bpy


def fbx_to_bvh(in_dir, out_dir = None):

    # in_dir = '<path to folder containing fbx file>'
    # out_dir = in_dir + '/fbx2bvh'  # in_dir.replace('fbx', 'bvh')

    if out_dir is None:
        out_dir = os.path.join(
                    os.path.dirname(in_dir), 'fbx2bvh')

    fbx_files = glob(osp.join(in_dir, '*.fbx'))

    pbar = tqdm(enumerate(fbx_files))
    for idx, in_file in pbar:

        pbar.set_description(f"{idx+1} / {len(fbx_files)} - {in_file}")

        in_file_no_path = osp.split(in_file)[1]
        motion_name = osp.splitext(in_file_no_path)[0]
        rel_in_file = osp.relpath(in_file, in_dir)
        rel_out_file = osp.join(osp.split(rel_in_file)[0], '{}'.format(motion_name), '{}.bvh'.format(motion_name))
        rel_dir = osp.split(rel_out_file)[0]
        out_file = osp.join(out_dir, rel_out_file)

        os.makedirs(osp.join(out_dir, rel_dir), exist_ok=True)

        bpy.ops.import_scene.fbx(filepath=in_file)

        action = bpy.data.actions[-1]

        # checking because of Kfir's code
        assert (action.frame_range[0] < 9999) \
            and (action.frame_range[1] > -9999), 
                f"action.frame_range = {action.frame_range} is out-of-range"  
        
        bpy.ops.export_anim.bvh(filepath=out_file,
                                frame_start=action.frame_range[0],
                                frame_end=action.frame_range[1], 
                                root_transform_only=True)
        bpy.data.actions.remove(bpy.data.actions[-1])


def mdm_to_bvh(input_bvh, motion_path):

    from src.mdm_syn.motion import BVH
    from src.mdm_syn.motion.InverseKinematics import animation_from_positions

    # input_bvh = '<the bvh on which your network has been trained>'
    # motion_path = '<the in_betweening output path>/results.npy'

    input_anim, names, _ = BVH.load(input_bvh)
    
    pos = np.load(motion_path, allow_pickle=True)
    pos = pos.item()['motion']
    pos = pos.transpose(0, 3, 1, 2) #     samples x joints x coord x frames 
                                    # ==> samples x frames x joints x coord
    parents = input_anim.parents  
    bvh_path = motion_path[:-4] + '_anim_{}.bvh'
    
    pbar = tqdm(enumerate(pos))
    for i, p in pbar:
        pbar.set_description(f'{i+1} / {len(pos)}')

        anim, sorted_order, _ = animation_from_positions(p, parents)
        BVH.save(bvh_path.format(i), anim, names=names)



