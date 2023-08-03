import os
import logging
import argparse
from datetime import datetime

import yaml
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to data root.')
    parser.add_argument('--config', type=str, help='Path to config yaml file.')
    parser.add_argument('--log_dir', type=str, default='results/', help='Path for logging and saving results.')
    parser.add_argument('--device_list', type=str, nargs='+', default='cuda:0', help='List of device(s) to use.')
    args = parser.parse_args()

    return args


def get_configs(path_or_args: [argparse.Namespace, str]) -> (dict, dict, dict, dict):

    if type(path_or_args) == argparse.Namespace:
        with open(path_or_args.config, 'r') as f:
            conf = yaml.load(f, yaml.Loader)
    else:
        with open(path_or_args, 'r') as f:
            conf = yaml.load(f, yaml.Loader)

    data_conf = conf['DATA']
    model_conf = conf['MODEL']
    try:
        diff_conf = conf['DIFFUSION']
    except KeyError:
        diff_conf = None
    train_conf = conf['TRAIN']

    return data_conf, model_conf, diff_conf, train_conf


def get_log_path(args) -> str:

    now = datetime.now()
    log_path = os.path.join(args.log_dir, f'{now.strftime("%Y.%m.%d %H_%M_%S")}/')
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'info.log'), encoding='utf-8', level=logging.INFO)

    return log_path


def dump_configs(data_conf: dict, model_conf: dict, diff_conf: dict, train_conf: dict, log_path: str):

    conf = {'DATA': data_conf, 'MODEL': model_conf, 'DIFFUSION': diff_conf, 'TRAIN': train_conf}

    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        yaml.dump(conf, f)


def extract_codes_from_dataloader(vqvae_model: nn.Module, dataloader) -> TensorDataset:
    """ Encode image inputs with vqvae and extract field of discrete latents
        (the embedding indices in the codebook with closest l2 distance)
    """

    import torch.multiprocessing
    #torch.multiprocessing.set_sharing_strategy('file_system')

    print("##### Extracting and saving latent codes...")
    device = next(vqvae_model.parameters()).device
    e1s, e2s, yps, yts = [], [], [], []
    # for sample, _, _, phase_label, _ in tqdm(dataloader):
    for sample, _, _, phase_label, tool_label in tqdm(dataloader):
        z_e = vqvae_model.encode(sample.to(device))
        # tuple of (bottom, top encoding indices) where each is (B,1,H,W)
        encoding_indices, _ = vqvae_model.quantize(z_e)
        e1, e2 = encoding_indices
        e1s.append(e1)
        e2s.append(e2)
        yps.append(phase_label)
        yts.append(tool_label)

    #torch.multiprocessing.set_sharing_strategy('file_descriptor')
    return TensorDataset(torch.cat(e1s).cpu(), torch.cat(e2s).cpu(), torch.cat(yps), torch.cat(yts))
