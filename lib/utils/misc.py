import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd

from lib.dataset.cataracts_dataset import CATARACTSDataset


def integer_mask_to_binary(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ Converts a (N, H, W) integer mask in [0, K-1]
        to a binary (N, K, H, W) mask in {0, 1}

    :param mask: Integer label mask in (N, H, W)
    :param num_classes: Amount of classes [K]
    :return: Binary label mask in (N, K, H, W)
    """

    binary_mask = torch.ones_like(mask)  # (N, H, W)
    binary_mask = binary_mask.unsqueeze(1).repeat(1, num_classes, 1, 1)  # (N, K, H, W)
    for k in range(num_classes):
        binary_mask[:, k, ...] = (mask == k).float()
    return binary_mask


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, phase_label=None, tool_label=None, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :],
                               condition=condition,
                               phase_label=phase_label,
                               tool_label=tool_label,
                               cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def label_names_to_vectors(dataset: CATARACTSDataset,
                           phase: str = None,
                           toolset: list[str] = None) -> (torch.Tensor, torch.Tensor):
    phase_label = None
    if phase is not None:
        phase_id = dataset.phase_label_names.index(phase)
        phase_label = torch.tensor([phase_id])
        # phase_label = F.one_hot(torch.tensor([phase_id]), num_classes=dataset.num_phases_classes)
        phase_label.unsqueeze(0)

    tool_label = None
    if toolset is not None:
        tool_ids = [dataset.tool_label_names.index(query_tool) for query_tool in toolset]
        tool_label = torch.zeros(size=(dataset.num_tool_classes,))
        for tool_id in tool_ids:
            tool_label[tool_id] = 1
        tool_label.unsqueeze(0)

    return phase_label, tool_label


def label_vectors_to_names(dataset: CATARACTSDataset,
                           phase: torch.Tensor = None,
                           toolset: torch.Tensor = None) -> (str, list[str]):
    phase_name = None
    if phase is not None:
        phase_id = phase.item()
        phase_name = dataset.phase_label_names[phase_id]

    tool_names = None
    if toolset is not None:
        tools_exist = np.where(toolset.cpu().numpy() == 1)[0]
        tool_names = [dataset.tool_label_names[tool_id] for tool_id in tools_exist]

    return phase_name, tool_names


def sample_conditioning_labels(ds: CATARACTSDataset,
                               phase_weight_path: str = 'phase_inv_sample_weights.npy',
                               top_table_path: str = 'top_table.csv',
                               num_samples: int = 1) -> (torch.Tensor, torch.Tensor):
    sampling_df = pd.read_csv(top_table_path, sep=";").set_index('Unnamed: 0')
    phase_sample_weights = np.load(phase_weight_path)
    phase_tensor = None
    toolset_tensor = None
    for n in range(num_samples):
        sampled_phase = np.random.choice(range(0, len(ds.phase_label_names)), size=(1,), p=phase_sample_weights)[0]
        phase_tensor = torch.tensor([sampled_phase]).view(1, 1) if phase_tensor is None \
            else torch.cat([phase_tensor, torch.tensor([sampled_phase]).view(1, 1)], dim=0)
        phase_name = ds.phase_label_names[sampled_phase]
        _df = sampling_df[phase_name]
        total_count = _df.loc['TotalCount']
        toolsets = []
        weights = []
        for row in range(len(_df)):
            toolset = _df.index[row]
            count = _df.iloc[row]
            if count > 0 and toolset != 'TotalCount':
                weights.append(total_count / count)
                toolsets.append(toolset)
        weights = np.array(weights) / sum(weights)
        toolsets = np.array(toolsets)

        sampled_toolset = np.random.choice(toolsets, size=(1,), p=weights)[0]
        sampled_toolset = [1 if tool in sampled_toolset else 0 for tool in ds.tool_label_names]
        toolset_tensor = torch.FloatTensor(sampled_toolset).view(1, -1) if toolset_tensor is None \
            else torch.cat([toolset_tensor, torch.FloatTensor(sampled_toolset).view(1, -1)], dim=0)

    return phase_tensor, toolset_tensor


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)
