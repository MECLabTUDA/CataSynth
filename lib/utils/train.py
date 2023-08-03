import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def set_grads(grads, params):
    for g,p in zip(grads, params):
        p.grad = g


def preprocess_latent(x, n_bits):
    """ preprosses discrete latents space [0, 2**n_bits) to model space [-1,1];
     if size of the codebook ie n_embeddings = 512 = 2**9 -> n_bit=9 """
    # 1. convert data to float
    # 2. normalize to [0,1] given quantization
    # 3. shift to [-1,1]
    return x.float().div(2 ** n_bits - 1).mul(2).add(-1)


def deprocess_latent(x, n_bits):
    """ deprocess x from model space [-1,1] to discrete latents space [0, 2**n_bits)
     where 2**n_bits is size of the codebook """
    # 1. shift to [0,1]
    # 2. quantize to n_bits
    # 3. convert data to long
    return x.add(1).div(2).mul(2 ** n_bits - 1).long()


def sample_prior(model, h, y, n_samples, input_dims, n_bits):
    model.eval()

    H, W = input_dims
    out = torch.zeros(n_samples, 1, H, W, device=next(model.parameters()).device)
    for hi in range(H):
        for wi in range(W):
            logits = model(out, y) if h is None else model(out, h, y)
            probs = F.softmax(logits, dim=1)
            sample = torch.multinomial(probs[:, :, :, hi, wi].squeeze(2), 1)
            # multinomial samples long tensor in [0, 2**n_bits), convert back to model space [-1,1]
            out[:, :, hi, wi] = preprocess_latent(sample, n_bits)
            del logits, probs, sample
    return deprocess_latent(out, n_bits)  # out (B,1,H,W) field of latents in latent space [0, 2**n_bits)


