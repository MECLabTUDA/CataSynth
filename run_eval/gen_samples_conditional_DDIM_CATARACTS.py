import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.dataset.cataracts_dataset import CATARACTSDataset
from lib.model.ddim import DDIM
from lib.model.conditional_unet import ConditionalUNet
from lib.utils.logging import save_images
from lib.utils.pre_train import get_configs
from lib.utils.misc import sample_conditioning_labels, label_vectors_to_names

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
diffusion_conf['NUM_SAMPLE_TIMESTEPS'] = 200

os.makedirs(args.log_path + "eval/gen_samples/", exist_ok=False)

DEV = args.device_list[0]
BATCH_SIZE = 64
STEPS = 30000//BATCH_SIZE
TARGET_SHAPE = (3, 270, 480)
diffusion_conf['NUM_SAMPLE_TIMESTEPS'] = 200  # / 1000 (DDPM)


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
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=8,
                     drop_last=True, shuffle=True, pin_memory=False)
print(f"{len(test_ds)} test samples")

#
# Model etc.
#
print("##### Loading model etc.")

diffusion = DDIM(diffusion_config=diffusion_conf, model_config=model_conf, device=DEV)
m = ConditionalUNet(data_config=data_conf,
                    model_config=model_conf,
                    diffusion_config=diffusion_conf,
                    num_phase_labels=test_ds.num_phases_classes,
                    num_semantic_labels=test_ds.num_tool_classes).to(DEV)
m = torch.nn.DataParallel(m, device_ids=args.device_list) if not args.device_list == ['cpu'] else m
m.load_state_dict(torch.load(args.log_path + "ckpt.pth", map_location='cpu')[0])
m.eval()

#
# Evaluation loop
#
print("##### Gnerating DDIM samples")

annotations = None

for i in tqdm(range(STEPS)):

    with torch.no_grad():

        if i == STEPS:
            break

        # Weighted sampling from the inverse of p(toolset,phase)
        phase_label, tool_label = sample_conditioning_labels(
            test_ds,
            num_samples=BATCH_SIZE
        )

        phase_label, tool_label = phase_label.long().to(DEV), tool_label.float().to(DEV)

        x = torch.randn(
            BATCH_SIZE,
            *eval(data_conf['SHAPE']),
            device=DEV,
        )

        gen_sample = diffusion.sample_image(x=x,
                                            model=m,
                                            guidance=True,
                                            guidance_strength=2.0,
                                            phase_label=phase_label,
                                            semantic_label=tool_label)
        N, C, H, W = gen_sample.shape

        gen_sample = F.interpolate(gen_sample, size=TARGET_SHAPE[1:], mode='bilinear')
        gen_sample = (gen_sample + 1.0) * 0.5
        gen_sample = (gen_sample * 255).type(torch.uint8).to(DEV)

        for n in range(N):
            total_sample_nr = i * BATCH_SIZE + n

            phase_name, tool_names = label_vectors_to_names(test_ds, phase_label[n], tool_label[n])
            phase_name = phase_name.replace("/", "")
            tool_names = [tool_name.replace("/", "") for tool_name in tool_names]

            save_images(
                gen_sample[n],
                os.path.join(args.log_path,
                             "eval/gen_samples/",
                             f"{phase_name}_{tool_names}_sample{total_sample_nr}.png")
            )
            label = np.concatenate([phase_label[n].cpu().numpy(), tool_label[n].cpu().numpy()], axis=-1)
            annotations = label if annotations is None else np.concatenate([annotations, label], axis=0)

np.save(os.path.join(args.log_path, "eval/gen_samples_annotations.npz"), annotations)
