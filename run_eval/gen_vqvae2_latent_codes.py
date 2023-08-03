import os
import argparse
from collections import namedtuple

import pickle
import lmdb
import torch
from tqdm import tqdm

from lib.model.vqvae2 import VQVAE
from lib.utils.pre_train import get_configs
from lib.utils.factory import get_CATARACTS_data

#
# Pre-eval
#

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to data root.')
parser.add_argument('--log_path', type=str, help='Path to log folder of pretrained model.')
parser.add_argument('--device_list', type=str, nargs='+', default='cuda:0', help='List of device(s) to use.')
args = parser.parse_args()

args.config = args.log_path + "config.yaml"
data_conf, model_conf, diffusion_conf, train_conf = get_configs(args)

os.makedirs(args.log_path + "eval/gen_samples/", exist_ok=True)

train_conf['BATCH_SIZE'] = 1
train_conf['VAL_SAMPLES'] = 1
DEV = args.device_list[0]
MAP_SIZE = 100 * 1024 * 1024 * 1024
NAME = 'latent_codes'

#
# Data
#
print("##### Loading data.")

train_ds, train_dl, val_ds, val_dl = get_CATARACTS_data(args, data_conf, train_conf)

#
# Model etc.
#
print("##### Loading model etc.")

m = VQVAE(
    in_channel=eval(data_conf['SHAPE'])[0],
    channel=model_conf['CHANNELS'],
    n_res_block=model_conf['N_RES_BLOCKS'],
    n_res_channel=model_conf['RES_CHANNELS'],
    embed_dim=model_conf['EMBED_DIM'],
    n_embed=model_conf['N_EMBEDDINGS'],
    decay=model_conf['EMA_DECAY']
).to(DEV)
m.load_state_dict(torch.load(args.log_path + "ckpt.pth", map_location='cpu')[0])
m.eval()

#
# Evaluation loop
#
print("##### Extracting VQ-VAE2 latent codes")

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'phase_label', 'tool_label', 'filename'])

#
# Train split
#

lmdb_env = lmdb.open(NAME + "_train", map_size=MAP_SIZE)
index = 0

with lmdb_env.begin(write=True) as txn:
    pbar = tqdm(train_dl)

    for img, _, filenames, y_phase, y_tool in pbar:
        img = img.to(DEV)

        _, _, _, id_t, id_b = m.encode(img)  # Get embeddings
        id_t = id_t.detach().cpu().numpy()
        id_b = id_b.detach().cpu().numpy()
        y_phase = y_phase.numpy()
        y_tool = y_tool.numpy()

        for file, yp, yt, top, bottom in zip(filenames, y_phase, y_tool, id_t, id_b):
            row = CodeRow(top=top, bottom=bottom, phase_label=yp, tool_label=yt, filename=file)
            txn.put(str(index).encode('utf-8'), pickle.dumps(row))
            index += 1
            pbar.set_description(f'inserted: {index}')

    txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))

#
# Val. split
#

index = 0
lmdb_env = lmdb.open(NAME + "_val", map_size=MAP_SIZE)

with lmdb_env.begin(write=True) as txn:
    pbar = tqdm(val_dl)

    for img, _, filenames, y_phase, y_tool in pbar:
        img = img.to(DEV)

        _, _, _, id_t, id_b = m.encode(img)
        id_t = id_t.detach().cpu().numpy()
        id_b = id_b.detach().cpu().numpy()
        y_phase = y_phase.numpy()
        y_tool = y_tool.numpy()

        for file, yp, yt, top, bottom in zip(filenames, y_phase, y_tool, id_t, id_b):
            row = CodeRow(top=top, bottom=bottom, phase_label=yp, tool_label=yt, filename=file)
            txn.put(str(index).encode('utf-8'), pickle.dumps(row))
            index += 1
            pbar.set_description(f'inserted: {index}')

    txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))
