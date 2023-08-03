import os
import logging
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def save_images(images: torch.Tensor, path: str, nrow: int = 8):
    if images.dim() == 4 and images.shape[0] == 1:
        images.squeeze_(0)
    grid = torchvision.utils.make_grid(images, nrow=nrow) if images.dim() > 3 else images
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()  # H, W, C
    im = Image.fromarray(ndarr)
    im.save(path)


#

def plot_grad_flow(named_parameters):
    """ Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow.

        Based on https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            n = n[:30] if len(n) > 30 else n
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    fig = plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    if len(layers) < 25:
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="r", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.autoscale()
    plt.tight_layout()

    return fig


def save_examples(example_tensor_list: list[torch.Tensor],
                  target_shape: tuple,
                  log_path: str,
                  epoch: int,
                  device: str,
                  nrow: int = 8,
                  names: list = None):

    if names is None:
        names = range(0, len(example_tensor_list))

    target_shape = tuple(target_shape)

    for i, example in enumerate(example_tensor_list):
        example = F.interpolate(example, size=target_shape[1:], mode='bilinear')
        example = (example + 1.0) * 0.5
        example = (example * 255).type(torch.uint8).to(device)
        save_images(example, os.path.join(log_path, f"epoch{epoch}_{names[i]}.png"), nrow)


def save_checkpoint(state_dict_list: list, log_path: str):
    torch.save(state_dict_list, os.path.join(log_path, "ckpt.pth"))

    val_log_str = f"\t{datetime.now().strftime('%H:%M:%S')}" \
                  f"\tCheckpoint saved."
    print(val_log_str)
    logging.info(msg=val_log_str)
