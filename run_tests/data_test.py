import os
import random
import unittest

import albumentations as A
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from lib.dataset.synthetic_dataset import SyntheticCATARACTSDataset
from lib.dataset.cataracts_dataset import CATARACTSDataset


class SyntheticDataTest(unittest.TestCase):

    def setUp(self) -> None:

        self.ds = SyntheticCATARACTSDataset(
            root=os.path.join(os.getcwd(), '../results/CATARACTS/2023.02.21 15_46_53/eval/'),
            resize_shape=(128, 128),
            normalize=(.5, .5),
        )

    def testTypes(self):
        img, _, _, phase_label, tool_label = self.ds[random.randint(a=0, b=len(self.ds))]
        assert type(img) == type(phase_label) == type(tool_label) == torch.Tensor

    def testShapes(self):
        img, _, _, phase_label, tool_label = self.ds[random.randint(a=0, b=len(self.ds))]
        print(img.shape)
        print(phase_label.shape)
        print(tool_label.shape)
        assert img.shape == torch.Size([3, 128, 128])
        assert phase_label.shape == torch.Size([1, ])
        assert tool_label.shape == torch.Size([CATARACTSDataset.num_tool_classes, ])


class CATARACTSDataTest(unittest.TestCase):

    @ classmethod
    def setUpClass(cls) -> None:
        cls.ds = CATARACTSDataset(
            root='/media/yannik/samsung_data_ssd_2/CATARACTS-videos-processed/',
            resize_shape=(128, 128),
            normalize=(.5, .5),
            mode='train',
            frame_step=5,
            n_seq_frames=1
        )
        print(str(len(cls.ds)) + " samples")

    def testTypes(self):
        img, _, file_name, phase_label, tool_label = self.ds[random.randint(a=0, b=len(self.ds))]
        # assert (type(file_name) == str or type(file_name) == list)
        assert type(img) == type(phase_label) == type(tool_label) == torch.Tensor

    def testShape(self):
        img, _, file_name, phase_label, semantic_label = self.ds[random.randint(a=0, b=len(self.ds))]
        assert img.shape == torch.Size([3, 128, 128])
        assert phase_label.shape == torch.Size([1])
        assert semantic_label.shape == torch.Size([21])

    def testSampling(self):
        dl = DataLoader(self.ds, batch_size=4, num_workers=1, drop_last=True, shuffle=True, pin_memory=False)
        for i, (_, _, file_name, _, _) in enumerate(dl):
            print(file_name)
            if i == 5:
                break
        print()
        for i, (_, _, file_name, _, _) in enumerate(dl):
            print(file_name)
            if i == 5:
                break

    def testToolWeightedSampling(self):
        self.ds.sample_img = False
        BS = 64
        dl = DataLoader(self.ds, batch_size=BS, num_workers=2, drop_last=True, shuffle=True, pin_memory=False)
        unweighted_tool_count = np.array([0.0]*self.ds.num_tool_classes)[np.newaxis, :].repeat(BS, axis=0)
        for i, (_, _, _, _, tool_label) in enumerate(tqdm(dl)):
            unweighted_tool_count = np.add(unweighted_tool_count, tool_label)
        print(unweighted_tool_count.mean(0)/i)
        sample_weights, _, _ = self.ds.get_tool_sample_weights()
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(self.ds), replacement=True)
        dl = DataLoader(self.ds, batch_size=BS, num_workers=2, drop_last=True, sampler=sampler, pin_memory=False)
        weighted_tool_count = np.array([0.0]*self.ds.num_tool_classes)[np.newaxis, :].repeat(BS, axis=0)
        for i, (_, _, _, _, tool_label) in enumerate(tqdm(dl)):
            weighted_tool_count = np.add(weighted_tool_count, tool_label)
        print(weighted_tool_count.mean(0) / i)

    def testCrossValidationFolds(self):
        train0, val0 = self.ds.get_fold(idx=0, n_folds=3)
        train1, val1 = self.ds.get_fold(idx=1, n_folds=3)
        train2, val2 = self.ds.get_fold(idx=2, n_folds=3)
        print(len(train0))
        print(len(train1))
        print(len(train2))
        print(len(val0))
        print(len(val1))
        print(len(val2))
        assert abs(len(train0) - len(train1)) <= 2
        assert abs(len(train0) - len(train2)) <= 2
        assert abs(len(train1) - len(train2)) <= 2
        assert abs(len(val0) - len(val1)) <= 2
        assert abs(len(val0) - len(val2)) <= 2
        assert abs(len(val1) - len(val2)) <= 2
        print(val0)
        print(val1)
        print(val2)


if __name__ == "__main__":
    unittest.main()
