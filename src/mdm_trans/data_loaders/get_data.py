from torch.utils.data import DataLoader

from src.mdm_trans.data_loaders.tensors import collate as all_collate
from src.mdm_trans.data_loaders.tensors import t2m_collate, t2m_transfer_collate


def get_dataset_class(name):
    if name == "humanml":
        from src.mdm_trans.data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, hml_mode='train', is_transfer_benchmark=False):
    if hml_mode == 'gt':
        from src.mdm_trans.data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate

    if name in ["humanml", "kit"]:
        return t2m_collate if not is_transfer_benchmark else t2m_transfer_collate
    else:
        return all_collate


def get_dataset(name, num_frames=-1, split='train', hml_mode='train', pose_rep='6d', 
                datapath=None, benchmark_path='', filter_path=''):

    DATA = get_dataset_class(name)
    kwargs = {}
    
    if datapath:
        kwargs.update({'datapath': datapath})
    if benchmark_path != '':
        kwargs.update({'benchmark_path': benchmark_path})
    if filter_path != '':
        kwargs.update({'filter_path': filter_path})

    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, mode=hml_mode, **kwargs)
    else:
        dataset = DATA(split=split, num_frames=num_frames, **kwargs)
    return dataset


def get_dataset_loader(name, batch_size, num_frames=-1, 
                       split='train', hml_mode='train', pose_rep='6d',
                       datapath=None, benchmark_path='', filter_path=''):
    dataset = get_dataset(name, num_frames, split, hml_mode, pose_rep=pose_rep, 
                          datapath=datapath, benchmark_path=benchmark_path, 
                                                filter_path=filter_path)
    loader = dataset_loader_from_data(dataset, name, batch_size)
    return loader


def dataset_loader_from_data(dataset, name, batch_size, num_workers=8):
    collate = get_collate_fn(name, getattr(dataset, 'hml_mode', None), 
            is_transfer_benchmark=getattr('dataset.opt', 'benchmark_path', '') != '')
    
    # avoid error when debugging a small dataset (e.g., to obtain overfit)
    corrected_batch_size = min(batch_size, len(dataset))

    loader = DataLoader(dataset, batch_size=corrected_batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=True, collate_fn=collate)
    return loader
