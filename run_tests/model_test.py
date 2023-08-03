import unittest

import yaml
import torch

from lib.model.unet import UNet
from lib.model.cgan import Generator, Discriminator


class UNetModelTest(unittest.TestCase):

    def setUp(self) -> None:
        self.device = 'cuda'
        with open('../lib/configs/ddpm_medium.yaml', 'r') as f:
            conf = yaml.load(f, yaml.Loader)
        data_conf = conf['DATA']
        model_conf = conf['MODEL']
        diffusion_conf = conf['DIFFUSION']
        self.data_shape = tuple(eval(data_conf['SHAPE']))
        self.model = UNet(data_conf, model_conf, diffusion_conf).to(self.device)

    def testShapes(self):
        fake_input = torch.rand(size=(4, *self.data_shape)).to(self.device)
        fake_timestamps = torch.ones(size=(4,)).to(self.device)
        with torch.no_grad():
            print(fake_input.shape)
            print(fake_timestamps.shape)
            prediction = self.model(fake_input, fake_timestamps)
        assert prediction.shape == fake_input.shape


class PhaseClassifierModelTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def testShapes(self):
        pass


class GAN_ModelTest(unittest.TestCase):

    def setUp(self) -> None:
        self.gen = Generator(latent_dim=100, label_embedding_dim=100)
        self.disc = Discriminator()

    def testGenShapes(self):
        fake_noise = torch.randn(size=(3, 100))
        pred = self.gen(fake_noise)
        assert pred.shape == torch.Size([3, 3, 128, 128])

    def testDiscShapes(self):
        fake_img = torch.randn(size=(3, 3, 128, 128))
        pred = self.disc(fake_img)
        assert pred.shape == torch.Size([3, 1])


class Conditional_GAN_ModelTest(unittest.TestCase):

    def setUp(self) -> None:
        self.gen = Generator(latent_dim=100, label_embedding_dim=100, n_phase_classes=3, n_tool_dims=3)
        self.disc = Discriminator(n_phase_classes=3, n_tool_dims=3)

    def testGenShapes(self):
        fake_noise = torch.randn(size=(3, 100))
        fake_phase = torch.tensor([[1], [0], [1]]).view(3, 1).long()
        fake_tools = torch.randint(low=0, high=1, size=(3, 3)).float()
        pred = self.gen(fake_noise, fake_phase, fake_tools)
        assert pred.shape == torch.Size([3, 3, 128, 128])

    def testDiscShapes(self):
        fake_img = torch.randn(size=(3, 3, 128, 128))
        fake_phase = torch.tensor([[1], [0], [1]]).view(3, 1).long()
        fake_tools = torch.randint(low=0, high=1, size=(3, 3)).float()
        pred = self.disc(fake_img, fake_phase, fake_tools)
        assert pred.shape == torch.Size([3, 1])
