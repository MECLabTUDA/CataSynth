import glob
import argparse

import timm
import torch
import torch.nn.functional as F
import albumentations as A
import numpy as np
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance,\
    LearnedPerceptualImagePatchSimilarity, InceptionScore
from tqdm import tqdm
from PIL import Image

from lib.utils.pre_train import get_configs
from lib.utils.misc import WrappedModel
from lib.dataset.cataracts_dataset import CATARACTSDataset


def eval_fid_kid(real_path: str, gen_path: str, device: str = 'cuda'):

    fid = FrechetInceptionDistance().to(device)
    kid = KernelInceptionDistance(normalize=True).to(device)

    resize = A.Resize(128, 128)

    real_imgs = glob.glob(real_path + "/train*/*.jpg")
    gen_imgs = glob.glob(gen_path + "*.png")

    for real_img_path in tqdm(real_imgs[::5]):
        # Forward real sample through the same pre-processing
        real = np.array(Image.open(real_img_path))
        real = resize(image=real)['image']
        real = torch.from_numpy(real) / 255.0
        real = TF.normalize(real, mean=[.5], std=[.5]).permute(2, 0, 1).unsqueeze(0).to(device)
        real = F.interpolate(real, size=(270, 480), mode='bilinear')
        real = (real + 1.0) * 0.5
        real = (real * 255).type(torch.uint8)
        fid.update(real, real=True)
        kid.update(real, real=True)

    for gen_img_path in tqdm(gen_imgs):
        gen = torch.from_numpy(np.array(Image.open(gen_img_path))).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        fid.update(gen, real=False)
        kid.update(gen, real=False)

    fid_score = fid.compute()
    kid_mean, kid_std = kid.compute()

    print(f"FID: {fid_score}")
    print(f"KID mean: {kid_mean} std: {kid_std}")


def eval_lpips_diversity(gen_path: str, device: str = 'cuda'):
    """
        Computes the mean and std LPIPS feature distance of generated images.
        The higher the score, the more diverse is the data.

        To reduce computational complexity, only a subset of images pairs is considered.
        Therefore, the list of images is randomly shuffled and pairs of images are drawn randomly from it.

    """

    lpips_score = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    to_tensor = tf.ToTensor()

    img_paths = glob.glob(gen_path + "*.png")
    np.random.shuffle(img_paths)

    feature_dists = []

    for i in tqdm(range(0, len(img_paths)-1)):
        img1 = to_tensor(Image.open(img_paths[i])).to(device).unsqueeze(0)
        img2 = to_tensor(Image.open(img_paths[i+1])).to(device).unsqueeze(0)
        feature_dists.append(lpips_score(img1, img2).item())

    print(f"LPIPS avg. dist {np.mean(feature_dists)} std {np.std(feature_dists)}")


def eval_is(gen_path: str, chckpt_path: str, device: str = 'cuda'):
    """
        Computes the inception score of generated images.

        Following https://arxiv.org/abs/1801.01973 we deploy
        a pre-trained tool-detection model's encoder.

    """
    img_paths = glob.glob(gen_path + "*.png")

    data_conf, model_conf, _, _ = get_configs(chckpt_path + "config.yaml")

    m = timm.create_model(model_conf['TYPE'].lower(),
                          pretrained=True,
                          num_classes=CATARACTSDataset.num_tool_classes).to(device)
    m = torch.nn.DataParallel(m, device_ids=[device]) if not device == 'cpu' else WrappedModel(m)
    try:
        m.load_state_dict(torch.load(chckpt_path + "ckpt.pth", map_location='cpu')[0])
    except:
        m.module.load_state_dict(torch.load(chckpt_path + "ckpt.pth", map_location='cpu')[0])
    # m.module.reset_classifier(0)
    m.eval()

    incep = InceptionScore(feature=m, splits=20).to(device)
    # incep = InceptionScore(num_features=2048, feature_extractor=m, device=device)
    to_tensor = tf.ToTensor()

    for img_path in tqdm(img_paths):
        img = to_tensor(Image.open(img_path)).to(device).unsqueeze(0)
        # img = torch.from_numpy(np.array(Image.open(img_path))).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        img = F.interpolate(img, size=eval(data_conf['SHAPE'])[1:])
        incep.update(img)

    try:
        mean_incep, std_incep = incep.compute()

        print(f"IS mean {mean_incep.item()} std {std_incep.item()}")
    except:
        mean_incep = incep.compute()

        print(f"IS mean {mean_incep}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=str, help='Path to real images')
    parser.add_argument('--gen', type=str, help='Path to generated images')
    parser.add_argument('--ft', type=str, help='Path to feature extractor')
    parser.add_argument('--device_list', nargs='+', default='cuda:0', help='List of device(s) to use.')
    args = parser.parse_args()

    eval_fid_kid(args.real, args.gen, args.device_list[0])
    eval_lpips_diversity(args.gen, args.device_list[0])
    eval_is(args.gen, args.ft, args.device_list[0])
