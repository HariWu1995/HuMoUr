from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate


def get_dataset_class(name):
    if name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12 import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    # elif name == "amass":
    #     from .amass import AMASS
    #     return AMASS
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset( name, num_frames, split: str = 'train', 
                                hml_mode: str = 'train', 
                            dataset_path: str = None, ):

    dataset_kwargs = dict(num_frames=num_frames, split=split)

    if name in ["humanml", "kit"]:
        dataset_kwargs.update(dict(mode=hml_mode))
    if dataset_path is not None:
        dataset_kwargs.update(dict(datapath=dataset_path))

    DATASET = get_dataset_class(name)
    dataset = DATASET(**dataset_kwargs)
    return dataset


def get_dataset_loader(  name, batch_size, num_frames, split: str = 'train', 
                                                    hml_mode: str = 'train', 
                                                dataset_path: str = None, ):

    dataset = get_dataset(name, num_frames, split, hml_mode, dataset_path)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=8, drop_last=True, collate_fn=collate,)
    return loader
