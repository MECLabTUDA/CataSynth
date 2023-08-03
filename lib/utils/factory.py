import albumentations as A
import torch.optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from lib.dataset.cataracts_dataset import CATARACTSDataset
from lib.dataset.synthetic_dataset import SyntheticCATARACTSDataset
from lib.dataset.lmdb_dataset import LMDB_Dataset
from lib.model.phase_classifier import PhaseClassifier, FullyConvClassifier


def get_CATARACTS_data(args, data_conf, train_conf):
    train_ds = CATARACTSDataset(
        root=args.data_path,
        normalize=eval(data_conf['NORM']) if data_conf['NORM'] is not None else None,
        resize_shape=eval(data_conf['SHAPE'])[-2:],
        crop_shape=eval(data_conf['CROP_DIM'])[-2:] if data_conf['CROP_DIM'] is not None else None,
        random_hflip=data_conf['RANDOM_H_FLIP'],
        random_brightness_contrast=data_conf['RANDOM_BRIGHTNESS_CONTRAST'],
        mode='train',
        frame_step=data_conf['FRAME_STEP'],
        n_seq_frames=data_conf['N_SEQ_FRAMES'] if data_conf['N_SEQ_FRAMES'] is not None else 1,
        overlapping_seq_chunks=data_conf['OVERLAPPING_CHUNKS'] if data_conf[
                                                                      'OVERLAPPING_CHUNKS'] is not None else False,
    )

    if data_conf['FOLD'] >= 0:
        train_ids, val_ids = train_ds.get_fold(idx=data_conf['FOLD'], n_folds=5)

        _train_ds = Subset(train_ds, indices=train_ids)
    else:
        _train_ds = train_ds
        train_ids = range(0, len(train_ds))

    sample_weights = None
    if data_conf['PHASE_WEIGHTED_SAMPLING']:
        sample_weights = _train_ds.dataset.get_phase_sample_weights()[0][train_ids].squeeze()
    elif data_conf['TOOL_WEIGHTED_SAMPLING']:
        sample_weights = _train_ds.dataset.get_tool_sample_weights()[0][train_ids].squeeze()

    # print(_train_ds.dataset.get_phase_sample_weights()[0][train_ids].squeeze().shape)
    if sample_weights is not None:

        if data_conf["N_SEQ_FRAMES"] > 1:
            sample_weights = sample_weights.mean(1)

        # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(_train_ds), replacement=False)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(_train_ds), replacement=True)

        train_dl = DataLoader(_train_ds, batch_size=train_conf['BATCH_SIZE'], num_workers=train_conf['NUM_WORKERS'],
                              sampler=sampler, drop_last=True, pin_memory=False)
    else:
        train_dl = DataLoader(_train_ds, batch_size=train_conf['BATCH_SIZE'], num_workers=train_conf['NUM_WORKERS'],
                              drop_last=True, shuffle=True, pin_memory=False)

    val_ds = CATARACTSDataset(
        root=args.data_path,
        normalize=eval(data_conf['NORM']) if data_conf['NORM'] is not None else None,
        resize_shape=eval(data_conf['SHAPE'])[-2:],
        crop_shape=eval(data_conf['CROP_DIM'])[-2:] if data_conf['CROP_DIM'] is not None else None,
        mode='val',
        frame_step=data_conf['FRAME_STEP'],
        n_seq_frames=data_conf['N_SEQ_FRAMES'] if data_conf['N_SEQ_FRAMES'] is not None else 1,
        overlapping_seq_chunks=data_conf['OVERLAPPING_CHUNKS'] if data_conf[
                                                                      'OVERLAPPING_CHUNKS'] is not None else False,
    )

    if data_conf['FOLD'] >= 0:
        _val_ds = Subset(train_ds, indices=val_ids)
    else:
        _val_ds = val_ds

    val_dl = DataLoader(_val_ds, batch_size=train_conf['VAL_SAMPLES'], num_workers=train_conf['NUM_WORKERS'],
                        sampler=None, drop_last=True, shuffle=True, pin_memory=False)

    return _train_ds, train_dl, _val_ds, val_dl


def get_LMBD_data(data_conf: dict, train_conf: dict):
    train_ds = LMDB_Dataset(data_conf['LMDB_TRAIN_DATA_PATH'])
    train_dl = DataLoader(train_ds,
                          batch_size=train_conf['BATCH_SIZE'],
                          num_workers=train_conf['NUM_WORKERS'],
                          drop_last=True, shuffle=True, pin_memory=False)
    val_ds = LMDB_Dataset(data_conf['LMDB_VAL_DATA_PATH'])
    val_dl = DataLoader(train_ds,
                        batch_size=train_conf['VAL_SAMPLES'],
                        num_workers=train_conf['NUM_WORKERS'],
                        drop_last=True, shuffle=True, pin_memory=False)

    return train_ds, train_dl, val_ds, val_dl


def get_synth_data(args, data_conf: dict, train_conf: dict):
    train_ds = SyntheticCATARACTSDataset(
        root=args.data_path,
        resize_shape=eval(data_conf['SHAPE'])[-2:],
        crop_shape=eval(data_conf['CROP_DIM'])[-2:] if data_conf['CROP_DIM'] is not None else None,
        normalize=eval(data_conf['NORM']) if data_conf['NORM'] is not None else None,
        random_hflip=data_conf['RANDOM_H_FLIP'],
        random_brightness_contrast=data_conf['RANDOM_BRIGHTNESS_CONTRAST'],
    )
    len(train_ds)

    if data_conf['FOLD'] >= 0:
        train_ids, val_ids = train_ds.get_fold(idx=data_conf['FOLD'], n_folds=5)
        _train_ds = Subset(train_ds, indices=train_ids)
    else:
        _train_ds = train_ds

    train_dl = DataLoader(_train_ds, batch_size=train_conf['BATCH_SIZE'], num_workers=train_conf['NUM_WORKERS'],
                          drop_last=True, shuffle=True, pin_memory=False)

    if data_conf['FOLD'] >= 0:
        _val_ds = Subset(train_ds, indices=val_ids)
    else:
        _val_ds = train_ds

    val_dl = DataLoader(_val_ds, batch_size=train_conf['VAL_SAMPLES'], num_workers=train_conf['NUM_WORKERS'],
                        drop_last=True, shuffle=True, pin_memory=False)

    return _train_ds, train_dl, _val_ds, val_dl


def get_classifier(model_type: str, img_size, num_phase_labels, num_tool_labels, fc_ch, dropout_p):
    if model_type == 'FullyConv':
        return FullyConvClassifier(img_size, num_phase_labels, num_tool_labels, fc_ch, dropout_p)
    elif model_type == 'DenseNet':
        return PhaseClassifier(img_size, num_phase_labels, num_tool_labels, fc_ch, dropout_p)
    else:
        raise ValueError("Unknown model type.")


def get_optimizer(params: list, train_conf: dict) -> torch.optim.Optimizer:
    if train_conf['OPTIM_TYPE'].upper() == 'MOMENTUMSGD':
        return torch.optim.SGD(params=params,
                               lr=train_conf['LR'],
                               momentum=train_conf['MOMENTUM'],
                               weight_decay=train_conf['WEIGHT_DECAY'])
    elif train_conf['OPTIM_TYPE'].upper() == 'RMSPROP':
        return torch.optim.RMSprop(params=params,
                                   lr=train_conf['LR'],
                                   momentum=train_conf['MOMENTUM'],
                                   weight_decay=train_conf['WEIGHT_DECAY'],
                                   eps=train_conf['EPS'])
    elif train_conf['OPTIM_TYPE'].upper() == 'ADAM':
        return torch.optim.Adam(params=params,
                                lr=train_conf['LR'],
                                betas=eval(train_conf['BETAS']),
                                weight_decay=train_conf['WEIGHT_DECAY'],
                                eps=train_conf['EPS'])
    elif train_conf['OPTIM_TYPE'].upper() == 'ADAMW':
        return torch.optim.AdamW(params=params,
                                 lr=train_conf['LR'],
                                 betas=eval(train_conf['BETAS']),
                                 weight_decay=train_conf['WEIGHT_DECAY'],
                                 eps=train_conf['EPS'])
    else:
        raise ValueError("Unknown optimizer type.")


def get_scheduler(optim: torch.optim.Optimizer, train_conf: dict):
    if train_conf['SCHED_TYPE'].upper() == 'POLYNOMIALLR':
        return torch.optim.lr_scheduler.PolynomialLR(optim,
                                                     total_iters=train_conf['EPOCHS'],
                                                     power=train_conf['SCHED_POWER'])
    elif train_conf['SCHED_TYPE'].upper() == 'REDUCELRONPLATEAU':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                          factor=train_conf['SCHED_FACTOR'],
                                                          patience=1)
    elif train_conf['SCHED_TYPE'].upper() == 'CONSTANTLR':
        return torch.optim.lr_scheduler.ConstantLR(optim)

    elif train_conf['SCHED_TYPE'].upper() == 'COSINEANNEALINGLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                          T_max=train_conf['EPOCHS'] // 4)

    elif train_conf['SCHED_TYPE'].upper() == 'EXPONENTIALLR':
        return torch.optim.lr_scheduler.ExponentialLR(optim,
                                                      gamma=train_conf['SCHED_FACTOR'])

    else:
        raise ValueError("Unknown LR scheduler type.")
