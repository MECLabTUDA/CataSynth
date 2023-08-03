import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from lib.model.vqvae2 import VQVAE
from lib.model.pixelsnail_prior import PixelSNAIL
from lib.dataset.cataracts_dataset import CATARACTSDataset
from lib.utils.logging import save_images
from lib.utils.pre_train import get_configs
from lib.utils.misc import sample_model, WrappedModel, sample_conditioning_labels, label_vectors_to_names

#
# Pre-eval
#

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to data root.')
parser.add_argument('--vqvae', type=str, help='Path to log folder of pretrained VQ-VAE2 model.')
parser.add_argument('--bottom', type=str, help='Path to log folder of pretrained bottom prior PixelSNAIL model.')
parser.add_argument('--top', type=str, help='Path to log folder of pretrained top prior PixelSNAIL model.')
parser.add_argument('--device_list', type=str, nargs='+', default='cuda:0', help='List of device(s) to use.')
args = parser.parse_args()

data_conf, vqvae_model_conf, _, _ = get_configs(args.vqvae + "config.yaml")
_, bottom_model_conf, _, _ = get_configs(args.bottom + "config.yaml")
_, top_model_conf, _, _ = get_configs(args.top + "config.yaml")

DEV = args.device_list[0]
BATCH_SIZE = 64
STEPS = 30000//BATCH_SIZE
TARGET_SHAPE = (3, 270, 480)
TARGET_PATH = 'results/new_vqvae_samples/'
os.makedirs(TARGET_PATH + "gen_samples/", exist_ok=False)
SHAPE = eval(data_conf['SHAPE'])
TEMP = 1.0

print(f"Avail. GPUs: ", torch.cuda.device_count())

#
# Data
#
print("##### Loading data.")

test_ds = CATARACTSDataset(
    root=args.data_path,
    resize_shape=eval(data_conf['SHAPE'])[1:],
    normalize=eval(data_conf['NORM']),
    mode='test',
    frame_step=data_conf['FRAME_STEP'],
    sample_img=False
)

test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4,
                     drop_last=True, shuffle=True, pin_memory=False)
print(f"{len(test_ds)} test samples")

#
# Model etc.
#
print("##### Loading model etc.")

vqvae = VQVAE(
    in_channel=eval(data_conf['SHAPE'])[0],
    channel=vqvae_model_conf['CHANNELS'],
    n_res_block=vqvae_model_conf['N_RES_BLOCKS'],
    n_res_channel=vqvae_model_conf['RES_CHANNELS'],
    embed_dim=vqvae_model_conf['EMBED_DIM'],
    n_embed=vqvae_model_conf['N_EMBEDDINGS'],
    decay=vqvae_model_conf['EMA_DECAY']
).to(DEV)
vqvae = torch.nn.DataParallel(vqvae, device_ids=args.device_list) if len(args.device_list) > 1 \
    else WrappedModel(vqvae)
try:
    vqvae.module.load_state_dict(torch.load(args.vqvae + "ckpt.pth", map_location='cpu')[0])
except:
    vqvae.load_state_dict(torch.load(args.vqvae + "ckpt.pth", map_location='cpu')[0])
vqvae.eval()

top_prior = PixelSNAIL(
    shape=[16, 16],
    n_class=top_model_conf['N_CLASS'],
    channel=top_model_conf['CHANNELS'],
    kernel_size=5,
    n_block=top_model_conf['N_BLOCKS'],
    n_res_block=top_model_conf['N_BLOCKS'],
    res_channel=top_model_conf['RES_CHANNELS'],
    dropout=top_model_conf['DROPOUT'],
    n_out_res_block=top_model_conf['N_OUT_RES_BLOCKS'],
    n_phase_labels=CATARACTSDataset.num_phases_classes,
    n_tool_labels=CATARACTSDataset.num_tool_classes,
    label_cond_ch=top_model_conf['LABEL_COND_CH']
).to(DEV)
top_prior = torch.nn.DataParallel(top_prior, device_ids=args.device_list) if len(args.device_list) > 1 \
    else WrappedModel(top_prior)
try:
    top_prior.module.load_state_dict(torch.load(args.top + "ckpt.pth", map_location='cpu')[0])
except Exception:
    top_prior.load_state_dict(torch.load(args.top + "ckpt.pth", map_location='cpu')[0])
top_prior.eval()

bottom_prior = PixelSNAIL(
    shape=[32, 32],
    n_class=bottom_model_conf['N_CLASS'],
    channel=bottom_model_conf['CHANNELS'],
    kernel_size=5,
    n_block=bottom_model_conf['N_BLOCKS'],
    n_res_block=bottom_model_conf['N_RES_BLOCKS'],
    res_channel=bottom_model_conf['RES_CHANNELS'],
    attention=False,
    dropout=bottom_model_conf['DROPOUT'],
    n_cond_res_block=bottom_model_conf['N_COND_RES_BLOCKS'],
    cond_res_channel=bottom_model_conf['RES_CHANNELS'],
    n_phase_labels=CATARACTSDataset.num_phases_classes,
    n_tool_labels=CATARACTSDataset.num_tool_classes,
    label_cond_ch=bottom_model_conf['LABEL_COND_CH']
).to(DEV)
bottom_prior = torch.nn.DataParallel(bottom_prior, device_ids=args.device_list) if len(args.device_list) > 1 \
    else WrappedModel(bottom_prior)
try:
    bottom_prior.module.load_state_dict(torch.load(args.bottom + "ckpt.pth", map_location='cpu')[0])
except Exception:
    bottom_prior.load_state_dict(torch.load(args.bottom + "ckpt.pth", map_location='cpu')[0])
bottom_prior.eval()

#
# Evaluation loop
#
print("##### Generating VQ-VAE2 samples...")

annotations = None

with torch.no_grad():

    for i in tqdm(range(STEPS)):

        # Weighted sampling from the inverse of p(toolset,phase)
        phase_label, tool_label = sample_conditioning_labels(
            test_ds,
            num_samples=BATCH_SIZE
            )
        phase_label, tool_label = phase_label.long().to(DEV), tool_label.float().to(DEV)

        top_sample = sample_model(top_prior.module,
                                  DEV,
                                  BATCH_SIZE,
                                  [16, 16],
                                  TEMP,
                                  phase_label=phase_label,
                                  tool_label=tool_label)
        bottom_sample = sample_model(bottom_prior.module,
                                     DEV,
                                     BATCH_SIZE,
                                     [32, 32],
                                     TEMP,
                                     phase_label=phase_label,
                                     tool_label=tool_label,
                                     condition=top_sample)

        decoded_sample = vqvae.module.decode_code(top_sample, bottom_sample)
        decoded_sample = decoded_sample.clamp(-1, 1)
        N, C, H, W = decoded_sample.shape

        gen_sample = F.interpolate(decoded_sample, size=TARGET_SHAPE[1:], mode='bilinear')
        gen_sample = (gen_sample + 1.0) * 0.5
        gen_sample = (gen_sample * 255).type(torch.uint8)

        for n in range(N):
            total_sample_nr = i * BATCH_SIZE + n
            phase_name, tool_names = label_vectors_to_names(test_ds, phase_label[n], tool_label[n])
            phase_name = phase_name.replace("/", "")
            tool_names = [tool_name.replace("/", "") for tool_name in tool_names]

            save_images(
                gen_sample[n],
                os.path.join(TARGET_PATH,
                             "gen_samples/",
                             f"{phase_name}_{tool_names}_sample{total_sample_nr}.png")
            )
            label = np.concatenate([phase_label[n].cpu().numpy(), tool_label[n].cpu().numpy()], axis=-1)
            annotations = label if annotations is None else np.concatenate([annotations, label], axis=0)

np.save(os.path.join(TARGET_PATH, "gen_samples_annotations.npy"), annotations)
