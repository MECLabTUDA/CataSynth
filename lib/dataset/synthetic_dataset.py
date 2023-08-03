import os
import glob
import random

import torch
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import albumentations.augmentations.functional as AF
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from natsort import natsorted
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm


class SyntheticCATARACTSDataset(Dataset):
    original_shape = (3, 1080, 1920)

    tool_label_names = ['Biomarker', 'Charleux Cannula', 'Hydrodissection Cannula', 'Rycroft Cannula',
                        'Viscoelastic Cannula', 'Cotton', 'Capsulorhexis Cystotome', 'Bonn Forceps',
                        'Capsulorhexis Forceps', 'Troutman Forceps',
                        'Needle Holder', 'Irrigation/Aspiration Handpiece', 'Phacoemulsifier Handpiece',
                        'Vitrectomy Handpiece', 'Implant Injector', 'Primary Incision Knife',
                        'Secondary Incision Knife', 'Micromanipulator', 'Suture Needle',
                        'Mendez Ring', 'Vannas Scissors']
    num_tool_classes = len(tool_label_names)

    phase_label_names = ['Idle', 'Toric Marking', 'Implant Ejection', 'Incision', 'Viscodilatation',
                         'Capsulorhexis', 'Hydrodissection', 'Nucleus Breaking', 'Phacoemulsification',
                         'Vitrectomy', 'Irrigation / Aspiration', 'Preparing Implant', 'Manual Aspiration',
                         'Implanting', 'Positioning', 'OVD Aspiration', 'Suturing', 'Sealing Control',
                         'Wound Hydration']
    num_phases_classes = len(phase_label_names)

    def __init__(self,
                 root: str,
                 resize_shape: tuple,
                 crop_shape: tuple = None,
                 normalize: tuple = None,
                 random_hflip: bool = False,
                 random_brightness_contrast: bool = False,
                 sample_img: bool = True,
                 remove_idle: bool = False
                 ):

        super(SyntheticCATARACTSDataset, self).__init__()

        assert os.path.isdir(root)
        assert os.path.isdir(root + 'gen_samples/')
        assert os.path.isfile(root + 'gen_samples_annotations.npz.npy')

        self.root = root
        self.sample_list = []
        self.sample_img = sample_img
        assert len(resize_shape) == 2
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.normalize = normalize
        self.random_hflip = random_hflip
        self.random_brightness_contrast = random_brightness_contrast
        self.remove_idle = remove_idle

        self.tool_counts = [0] * self.num_tool_classes
        self.tool_counts = np.array(self.tool_counts)

        self.phase_counts = [0] * self.num_phases_classes
        self.phase_counts = np.array(self.phase_counts)

        # Read annotations
        self.annotations = np.load(root + 'gen_samples_annotations.npz.npy')
        self.annotations = np.reshape(self.annotations, (-1, 22))
        print('annotations ', self.annotations.shape)

        for img_file_path in natsorted(glob.glob(os.path.join(root, 'gen_samples/*.png'))):

            self.sample_list.append(img_file_path)

        self.dataset = self

    def get_fold(self, idx: int, n_folds: int = 5):
        all_ids = np.arange(0, len(self))
        splits = np.array_split(all_ids, indices_or_sections=n_folds)
        val_ids = splits[idx]
        train_ids = np.concatenate([splits[i] for i in np.delete(np.arange(0, n_folds), idx)])
        return train_ids, val_ids

    def transform(self, img_list: list[np.ndarray]) -> torch.Tensor:

        aug_list = [A.Resize(*self.resize_shape)]

        if self.crop_shape is not None:
            aug_list.append(A.RandomCrop(*self.crop_shape, always_apply=True))

        if self.random_hflip:
            aug_list.append(A.HorizontalFlip(p=.3))

        if self.random_brightness_contrast:
            aug_list.append(A.RandomBrightnessContrast(p=.3))

        transf = A.Compose(aug_list,
                           additional_targets={'image' + str(key): 'image' for key in range(0, len(img_list) - 1)})

        kwargs = {}
        kwargs['image'] = img_list[0]
        for key in range(0, len(img_list) - 1):
            kwargs['image' + str(key)] = img_list[key + 1]

        data = transf(**kwargs)

        img_tensor = []
        for str_key in kwargs.keys():
            img = data[str_key]
            img = torch.from_numpy(img) / 255.0
            if self.normalize is not None:
                TF.normalize(img, mean=self.normalize[0], std=self.normalize[1], inplace=True)
            img_tensor.append(img.permute(2, 0, 1).unsqueeze(0))  # T, C, H, W
        img_tensor = torch.cat(img_tensor, dim=0)

        return img_tensor

    def __getitem__(self, index: int):
        img_file_path = self.sample_list[index]

        # name = img_file_path.split("/")[-1]

        sample_nr = img_file_path.split("_")[-1].removeprefix('sample').removesuffix('.png')
        sample_nr = int(sample_nr)

        img = np.array(Image.open(img_file_path)) if self.sample_img else np.zeros(1)
        img_tensor = self.transform([img]).squeeze(0)  # C, H, W

        # Phase to integer
        phase_label_tensor = torch.tensor([self.annotations[sample_nr, 0]]).long()  # 1,

        # Tool to one-hot
        tools = np.array(self.annotations[sample_nr, 1:])
        tool_label_tensor = torch.tensor(tools).squeeze(0)  # K,

        assert not torch.isnan(phase_label_tensor).any()
        assert not torch.isnan(tool_label_tensor).any()

        #print(f'synth phase: type: {phase_label_tensor.dtype} shape: {phase_label_tensor.shape}')
        #print(f'synth tools: type: {tool_label_tensor.dtype} shape: {tool_label_tensor.shape}')

        # return img_tensor, torch.zeros(size=(1, 1)), [sample_nr], phase_label_tensor, tool_label_tensor
        return img_tensor, torch.zeros(size=(1, 1)), torch.zeros(size=(1, 1)), phase_label_tensor, tool_label_tensor

    def __len__(self):
        return len(self.sample_list)

    def __dataset__(self):
        return self
