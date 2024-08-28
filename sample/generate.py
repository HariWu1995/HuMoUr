from src.mdm.generate import main as generate_pipe
from src.mdm.utils.parser_util import generate_args


def main():
    args = generate_args()
    
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    out_path = args.output_dir
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    all_motions, all_text, all_lengths = generate_pipe(args)

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

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption =   all_text[rep_i*args.batch_size + sample_i]
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))

            animation_save_path = os.path.join(out_path, save_file)
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
            
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(args, out_path,
                                            row_print_template, all_print_template, 
                                            row_file_template, all_file_template,
                                            caption, num_samples_in_out_file, 
                                            rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
