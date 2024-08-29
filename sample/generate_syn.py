import os
import shutil
import copy
import numpy as np

from src.mdm_syn.generate import main as generate_pipe
from src.mdm_syn.utils.parser_util import generate_args

from src.mdm_syn.data_loaders.humanml.utils.plot_script import plot_3d_motion


def main():
    args = generate_args()

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    
    out_path = args.output_dir
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    setattr(args, 'out_path', out_path)
    all_motions, all_text, all_lengths, \
    data, model_kwargs, skeleton, n_joints, fps = generate_pipe(args)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {  'motion': all_motions, 
                           'text': all_text, 
                        'lengths': all_lengths,
                    'num_samples': args.num_samples,})

    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))

    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
     sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        caption  =  all_text[0 * args.batch_size + sample_i]
        length = all_lengths[0 * args.batch_size + sample_i]
        motion = all_motions[0 * args.batch_size + sample_i].transpose(2, 0, 1)[:length]  # n_joints x 3 x n_frames  ==>  n_frames x n_joints x 3
        
        save_file = sample_file_template.format(sample_i, 0 )
        print(sample_print_template.format(caption, sample_i, 0 , save_file))

        animation_save_path = os.path.join(out_path, save_file)
        motion_to_plot = copy.deepcopy(motion)

        if 'Breakdancing_Dragon' in args.sin_path:
            motion_to_plot = motion_to_plot[:, :, [0,2,1]] # swap y and z axes

        # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
        plot_3d_motion(animation_save_path, skeleton, motion_to_plot, 
                        dataset=args.dataset, title=caption, fps=fps)

        rep_files.append(animation_save_path)

        sample_files, all_sample_save_path = save_multiple_samples(
            args, out_path,
            row_print_template, all_print_template, 
            row_file_template, all_file_template,
            caption, num_samples_in_out_file, 
            rep_files, sample_files, sample_i
        )

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

    assert all_sample_save_path is not None
    return all_sample_save_path


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)

    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    all_sample_save_path = None

    if (sample_i + 1) % num_samples_in_out_file == 0 \
    or (sample_i + 1 == args.num_samples):
        # save several samples together
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'

        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files, all_sample_save_path


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'

    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


if __name__ == "__main__":
    main()
