import os
import shutil
import numpy as np

from src.mdm.edit import main as edit_pipe, recover_from_ric
from src.mdm.utils.parser_util import edit_args

from src.mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
import src.mdm.data_loaders.humanml.utils.paramUtil as paramUtil


def main():
    args = edit_args()

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    
    out_path = args.output_dir
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.edit_mode, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    all_motions, all_text, all_lengths, \
        data, model_kwargs, \
        gt_frames_per_sample, n_joints, fps = edit_pipe(args)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {  'motion': all_motions, 
                           'text': all_text, 
                        'lengths': all_lengths,
                    'num_samples': args.num_samples, 
                'num_repetitions': args.num_repetitions, })

    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))

    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else \
               paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

    for sample_i in range(args.num_samples):
        caption = 'Input Motion'
        input_lengths = model_kwargs['y']['lengths']
        length = input_lengths[sample_i]
        motion = input_motions[sample_i].transpose(2, 0, 1)[:length]

        save_file = 'input_motion{:02d}.mp4'.format(sample_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')

        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                       dataset=args.dataset, fps=fps, vis_mode='gt',
                       gt_frames=gt_frames_per_sample.get(sample_i, []))

        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i * args.batch_size + sample_i]
            if caption == '':
                caption = 'Edit [{}] unconditioned'.format(args.edit_mode)
            else:
                caption = 'Edit [{}]: {}'.format(args.edit_mode, caption)
                
            length = all_lengths[rep_i * args.batch_size + sample_i]
            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            
            # Credit for visualization: 
            #   https://github.com/EricGuo5513/text-to-motion
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                           dataset=args.dataset, fps=fps, vis_mode=args.edit_mode,
                           gt_frames=gt_frames_per_sample.get(sample_i, []))

        all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions+1}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
