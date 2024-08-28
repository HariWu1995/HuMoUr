from src.mdm_prior.sequentialize import main as generate_pipe
from src.mdm_prior.utils.parser_util import generate_args


def calc_frame_colors(handshake_size, blend_size, step_sizes, lengths):
    for ii, step_size in enumerate(step_sizes):
        if ii == 0:
            frame_colors = ['orange'] * (step_size - handshake_size - blend_size) + \
                           ['blue'] * blend_size + \
                           ['purple'] * (handshake_size // 2)
            continue
        if ii == len(step_sizes) - 1:
            frame_colors += ['purple'] * (handshake_size // 2) + \
                            ['blue'] * blend_size + \
                            ['orange'] * (lengths[ii] - handshake_size - blend_size)
            continue
        frame_colors += ['purple'] * (handshake_size // 2) + ['blue'] * blend_size + \
                        ['orange'] * (lengths[ii] - 2 * handshake_size - 2 * blend_size) + \
                        ['blue'] * blend_size + \
                        ['purple'] * (handshake_size // 2)
    return frame_colors


def main():
    print(f"generating samples")
    args = generate_args()

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    out_path = args.output_dir
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'DoubleTake_samples_{}_{}_seed{}'.format(name, niter, args.seed))
        
        if args.input_text != '':
            if ".txt" in args.input_text:
                out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
                is_txt = True
            elif ".csv" in args.input_text:
                out_path += '_' + os.path.basename(args.input_text).replace('.csv', '').replace(' ', '_').replace('.', '')
                is_csv = True
            else:
                raise TypeError("Incorrect text file type, use csv or txt")

        if args.sample_gt:
            out_path += "_gt"
        out_path += f"_handshake_{args.handshake_size}"

        if args.double_take:
            out_path += "_double_take"
            out_path += f"_blend_{args.blend_len}"
            out_path += f"_skipSteps_{args.skip_steps_double_take}"

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    all_motions, all_text, all_lengths, y_lengths = generate_pipe(args)
    frame_colors = calc_frame_colors(args.handshake_size, args.blend_len, step_sizes, y_lengths)
    
    npy_path = os.path.join(out_path, 'results.npy')

    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {  'motion': all_motions, 
                           'text': all_text, 
                        'lengths': all_lengths,
                    'num_samples': args.num_samples, 
                'num_repetitions': num_repetitions, 
                   'frame_colors': frame_colors,    })
    
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else \
               paramUtil.t2m_kinematic_chain
    # if args.dataset == 'babel':
    #     skeleton = paramUtil.t2m_kinematic_chain

    sample_files = []
    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i, samples_type_i in zip(range(num_repetitions), samples_type):
            caption = [f'{samples_type_i} {all_text[0]}'] * (y_lengths[0] - int(args.handshake_size/2))
            for ii in range(1, old_num_samples):
                caption += [f'{samples_type_i} {all_text[ii]}'] * (int(y_lengths[ii])-args.handshake_size)
            
            caption += [f'{samples_type_i} {all_text[ii]}'] * (int(args.handshake_size/2))
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            print(f'[({sample_i}) "{set(caption)}" | Rep #{rep_i} | -> {save_file}]')
            
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps,
                           vis_mode='gt' if args.sample_gt else 'unfold_arb_len', handshake_size=args.handshake_size,
                           blend_size=args.blend_len,step_sizes=step_sizes, lengths=y_lengths)
            
            # Credit for visualization: 
            #       https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)
        
        if num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={num_repetitions}' if num_repetitions > 1 else ''
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)

            print(f'[({sample_i}) "{set(caption)}" | all repetitions | -> {all_rep_save_file}]')
            sample_files.append(all_rep_save_file)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
