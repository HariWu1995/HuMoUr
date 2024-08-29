import os
import shutil
import numpy as np

from src.mdm_prior.complete import main as edit_pipe
from src.mdm_prior.utils.parser_util import edit_multi_args

import src.mdm_prior.data_loaders.humanml.utils.paramUtil as paramUtil
from src.mdm_prior.data_loaders.humanml.utils.plot_script import plot_3d_motion


def main():

    args = edit_multi_args()

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    out_path = args.output_dir
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'prefix_completion_{}_{}_{}_seed{}'.format(name, niter, args.edit_mode, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    all_motions, all_text, all_lengths, sample, sample1, \
     data, model_kwargs, gt_save, gt_frames_per_sample, fps = edit_pipe(args)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {  'gt': gt_save, 
                     'motion': all_motions, 
                       'text': all_text, 
                    'lengths': all_lengths,
                'num_samples': args.num_samples, 
            'num_repetitions': args.num_repetitions, })
             # 'all_captions': all_captions, })

    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))

    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else \
               paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 7
    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = 'Prefix Completion'
            length = all_lengths[rep_i * args.batch_size + sample_i] - 1
            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            motion1 = None
            if sample1 is not None:
                motion1 = sample1[sample_i].transpose(2, 0, 1)[:length]

            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')

            # Credit for visualization: 
            #       https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)
            plot_3d_motion(animation_save_path, skeleton, motion, 
                           dataset=args.dataset, title=caption, fps=fps,
                           vis_mode=args.edit_mode, joints2=motion1, 
                           gt_frames=gt_frames_per_sample.get(sample_i, []))
                           #, captions=captions)

        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)

            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')
            sample_files.append(all_rep_save_file)

            if (sample_i+1) % num_samples_in_out_file == 0 \
            or (sample_i+1 == args.num_samples):
                all_sample_save_file = os.path.join(out_path, f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4')
                ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
                vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
                ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_file}'
                os.system(ffmpeg_rep_cmd)

                print(f'[(samples {(sample_i - len(sample_files) + 1):02d} to {sample_i:02d}) | all repetitions | -> {all_sample_save_file}]')
                sample_files = []

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
