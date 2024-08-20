from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from src.mdm_prior.data_loaders.tensors import collate as all_collate, babel_collate, babel_eval_collate, pw3d_collate
from src.mdm_prior.data_loaders.tensors import t2m_collate
from src.mdm_prior.data_loaders.amass.sampling import FrameSampler
from src.mdm_prior.data_loaders.humanml.data.dataset import collate_fn as sorted_collate


def get_dataset_class(name, load_mode):

    if name == "babel":
        if load_mode == "text_only":
            load_mode = 'train'

        if load_mode in ['gt', 'eval', 'movement_train', 'evaluator_train']:
            from src.mdm_prior.data_loaders.humanml.data.dataset import BABEL_eval
            return BABEL_eval

        elif load_mode == 'train':
            from src.mdm_prior.data_loaders.amass.babel import BABEL
            return BABEL

        else:
            raise ValueError(f'Unsupported load_moad name [{load_mode}]')

    elif name == "humanml":
        from src.mdm_prior.data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D

    elif name == "pw3d":
        from src.mdm_prior.data_loaders.humanml.data.dataset import PW3D
        return PW3D

    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, load_mode='train'):

    if load_mode == 'gt':
        from src.mdm_prior.data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    
    if name in ["humanml", "kit"]:
        return t2m_collate
    
    elif name == 'pw3d':
        return pw3d_collate
    
    elif name == 'babel':
        
        if load_mode =='eval':
            return babel_eval_collate
        elif load_mode == 'movement_train':
            return default_collate
        elif load_mode == 'evaluator_train':
            return sorted_collate
        else:
            return babel_collate

    else:
        return all_collate


def get_dataset(name, num_frames, split='train', load_mode='train', 
                batch_size=None, opt=None, short_db=False, 
                cropping_sampler=False, size=None,  
                dataset_path: str = None, ):

    dataset_kwargs = dict(split=split)

    if name in ["humanml", "pw3d"]:
        dataset_kwargs.update(dict(num_frames=num_frames, load_mode=load_mode, size=size))
    
    elif name == "babel":
        from data_loaders.amass.transforms import SlimSMPLTransform

        if (split == 'val') and (cropping_sampler == True):
            transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True, canonicalize=False)
        else:
            transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)

        sampler = FrameSampler(min_len=num_frames[0], 
                               max_len=num_frames[1])
        dataset_kwargs.update(dict(transforms=transform, load_mode=load_mode, mode='train', opt=opt, 
                                    short_db=short_db, sampler=sampler, cropping_sampler=cropping_sampler))
    else:
        dataset_kwargs.update(dict(num_frames=num_frames))
        
    if dataset_path is not None:
        dataset_kwargs.update(dict(datapath=dataset_path))

    DATASET = get_dataset_class(name, load_mode)
    dataset = DATASET(**dataset_kwargs)

    return dataset


def get_dataset_loader(name, batch_size, num_frames, 
                        split='train', load_mode='train', opt=None, 
                        short_db=False, cropping_sampler=False, size=None, 
                        dataset_path: str = None):

    if load_mode == 'text_only':
        load_mode = 'train'

    collate = get_collate_fn(name, load_mode)
    dataset = get_dataset(name, num_frames, split, load_mode, batch_size, opt, 
                          short_db, cropping_sampler, size, dataset_path=dataset_path)

    n_workers = 1 if load_mode in ['movement_train', 'evaluator_train'] else 8
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, drop_last=True, collate_fn=collate,
    )

    return dataloader