from src.mdm_prior.control import main as edit_inpainting_pipe
from src.mdm_prior.utils.parser_util import edit_inpainting_args


def main():
    args_list = edit_inpainting_args()
    args = args_list[0]

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    out_path = args.output_dir
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    input_motions, y_lengths, \
      all_motions, all_text, all_lengths = edit_inpainting_pipe(args_list)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {'motion': all_motions, 
                         'text': all_text, 
                      'lengths': all_lengths,
                  'num_samples': args.num_samples, 
              'num_repetitions': args.num_repetitions})

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

    gt_frames_per_sample = {}
    for sample_i in range(args.num_samples):
        rep_files = []
        if args.show_input:
            caption = 'Input Motion'
            length =     y_lengths[sample_i]
            motion = input_motions[sample_i].transpose(2, 0, 1)[:length]

            save_file = 'input_motion{:02d}.mp4'.format(sample_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)

            print(f'[({sample_i}) "{caption}" | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                            dataset=args.dataset, fps=fps, vis_mode='gt',
                            gt_frames=gt_frames_per_sample.get(sample_i, []))

        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            if args.guidance_param == 0:
                caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
            else:
                caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
                
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)

            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                           dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
                           gt_frames=gt_frames_per_sample.get(sample_i, []), painting_features=args.inpainting_mask.split(','))
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + (1 if args.show_input else 0)} '
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)

            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()