import os
import shutil
import numpy as np
from typing import List, Tuple

from src.mdm_trans.transfer import main as transfer_pipe
from src.mdm_trans.utils.parser_util import transfer_args
from src.mdm_trans.utils.visualize import plot_3d_motion, save_multiple_samples

import src.mdm_trans.data_loaders.humanml.utils.paramUtil as paramUtil


def main():

    args = transfer_args()

    all_motions, all_text, all_lengths, \
    out_path, fps, max_frames, transfer_idx, n_followers = transfer_pipe(args)

    # save outputs in an xyz format
    save_results(args, out_path, all_motions, all_text, all_lengths)
    visualize_motions(args, out_path, all_motions, all_text, all_lengths, 
                       fps, max_frames, transfer_idx, n_followers)

    return out_path


def save_results(args, out_path, all_motions, all_text, all_lengths):
    assert all_motions.shape[-2] == 3, \
        f"Expected 3 channels for XYZ, but got {all_motions.shape[-2]}"
    
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {  'motion': all_motions, 
                           'text': all_text, 
                        'lengths': all_lengths,
                    'num_samples': args.num_samples, 
                'num_repetitions': args.num_repetitions,})

    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)


def visualize_motions(args, out_path: List[str], all_motions: List[np.ndarray],
                      all_text: List[List[str]], all_lengths: List[np.ndarray],
                      fps: float, max_frames: int, transfer_idx: dict, n_follower: int):
    
    print(f"saving visualizations to {out_path}...")
    
    # get kinematic chain
    kinematic_chain = paramUtil.t2m_kinematic_chain

    n_follower_mult = len(transfer_idx['follower'])
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    anim_leader_and_out = np.empty(shape=(args.num_repetitions, n_follower + 1 ), dtype=object)
    anim_follower       = np.empty(shape=(args.num_repetitions, n_follower_mult), dtype=object)

    idx_sample_i = 0
    for rep_i in range(args.num_repetitions):

        # plot leader motion
        anim_leader_and_out[rep_i, transfer_idx['leader']] = \
               prepare_plot(rep_i, transfer_idx['leader'], args, fps, all_motions, all_text, all_lengths, kinematic_chain, crop=True)
        
        # plot output motion
        for sample_i in transfer_idx['out']:
            # plot transferred motion near leader
            anim_leader_and_out[rep_i, sample_i-n_follower_mult] = \
                   prepare_plot(rep_i, sample_i, args, fps, all_motions, all_text, all_lengths, kinematic_chain, crop=True)
        
        # plot follower motion in a separate figure
        for sample_i in transfer_idx['follower']:
            anim_follower[rep_i, sample_i-1] = \
             prepare_plot(rep_i, sample_i, args, fps, all_motions, all_text, all_lengths, kinematic_chain)  # we do not crop the follower because there could be multiple followers with varying lengths
                
        idx_sample_i += args.num_samples

    # save_multiple_samples should be called outside the args loop
    save_multiple_samples(out_path, {'all': all_file_template}, anim_leader_and_out, fps, all_lengths[0], prefix='transfer_', n_rows_in_out_file=args.n_rows_in_out_file)
    save_multiple_samples(out_path, {'all': all_file_template}, anim_follower      , fps,     max_frames, prefix='follower_', n_rows_in_out_file=args.n_rows_in_out_file)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at {abs_path}')

    return out_path


def prepare_plot(rep_i, sample_i, args, fps, 
                 all_motions, all_text, all_lengths, 
                 kinematic_chain, crop=False):
    
    assert all_motions.shape[-2] == 3, \
        f"Expected 3 channels for XYZ, but got {all_motions.shape[-2]}"
    
    length = all_lengths[rep_i * args.batch_size + sample_i]
    motion = all_motions[rep_i * args.batch_size + sample_i]
    caption  =  all_text[rep_i * args.batch_size + sample_i]

    motion = motion.transpose(2, 0, 1)
    if crop:
        motion = motion[:length]
    else:
        # duplicate the last frame to end of motion, so all motions will be in equal length
        motion[length:-1] = motion[length-1]  
        
    plot = plot_3d_motion(kinematic_chain, motion, dataset=args.dataset, title=caption, fps=fps)
    return plot


if __name__ == "__main__":
    main()
