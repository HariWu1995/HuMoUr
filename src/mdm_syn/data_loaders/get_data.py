from torch.utils.data import DataLoader

from src.mdm_syn.data_loaders.tensors import collate as all_collate
from src.mdm_syn.data_loaders.tensors import t2m_collate


def get_dataset_class(name):
    if name == "humanml":
        from src.mdm_syn.data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, hml_mode='train'):

    if hml_mode == 'gt':
        from src.mdm_syn.data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate

    if name == "humanml":
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, 
                split='train', hml_mode='train', 
                dataset_path: str = None, ):
                
    dataset_kwargs = dict(num_frames=num_frames, split=split)

    if name in ["humanml"]:
        dataset_kwargs.update(dict(mode=hml_mode))
    if dataset_path is not None:
        dataset_kwargs.update(dict(datapath=dataset_path))

    DATASET = get_dataset_class(name)
    dataset = DATASET(**dataset_kwargs)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, 
                        split='train', hml_mode='train',
                        dataset_path: str = None, ):

    dataset = get_dataset(name, num_frames, split, hml_mode, dataset_path)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=8, drop_last=True, collate_fn=collate,)
    return loader