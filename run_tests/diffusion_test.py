import unittest

import torch
import yaml

from lib.model.ddim import DDIM
from lib.model.unet import UNet


class DiffusionTest(unittest.TestCase):

    def setUp(self) -> None:

        self.device = 'cuda'
        with open('../lib/configs/ddim_tiny.yaml', 'r') as f:
            conf = yaml.load(f, yaml.Loader)
        data_conf = conf['DATA']
        model_conf = conf['MODEL']
        diffusion_conf = conf['DIFFUSION']

        self.data_shape = tuple(eval(data_conf['SHAPE']))
        self.diffusion = DDIM(diffusion_conf, model_conf, self.device)
        self.m = UNet(data_conf, model_conf, diffusion_conf).to(self.device)
        self.m.eval()

    def testSampling(self):
        x = torch.randn(
            8,
            *self.data_shape,
            device=self.device,
        )
        with torch.no_grad():
            out = self.diffusion.sample_image(x=x, model=self.m)
        assert tuple(out.shape) == (8, *self.data_shape)
