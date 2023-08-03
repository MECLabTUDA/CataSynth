import os
import torch
import torch.nn.functional as F

from tqdm import tqdm

from lib.utils.factory import get_CATARACTS_data
from lib.utils.pre_train import parse_args, get_configs

#
# Pre-train
#
print("##### Pre-train")

args = parse_args()

data_conf, model_conf, diffusion_conf, train_conf = get_configs(args)

#
# Data
#
print("##### Loading data.")

train_ds, train_dl, val_ds, val_dl = get_CATARACTS_data(args, data_conf, train_conf)

#
# Weights claculation
#
print("##### Calculating weights.")
weight_per_tool_class = torch.zeros(train_ds.dataset.num_tool_classes)
weight_per_phase_class = torch.zeros(train_ds.dataset.num_phases_classes)
count_all_samples = torch.zeros(1)

for i, (img, _, _, phase_label, semantic_label) in enumerate(tqdm(train_dl)):
    count_all_samples += semantic_label.size(0)*semantic_label.size(1)
    weight_per_phase_class += F.one_hot(phase_label, num_classes=train_ds.dataset.num_phases_classes).sum(0).sum(0).sum(0)
    weight_per_tool_class += semantic_label.sum(0).sum(0)
weight_per_phase_class = weight_per_phase_class.float() / count_all_samples.float()
weight_per_tool_class = weight_per_tool_class.float() / count_all_samples.float()

#
# Saving
#
print("Phase weights", weight_per_phase_class)
print("Tool weights", weight_per_tool_class)

print("##### Saving weights.")
torch.save({"n_all_samples": count_all_samples,
            "phase": weight_per_phase_class,
            "tool": weight_per_tool_class}, "lib/configs/weights_per_class.pt")
