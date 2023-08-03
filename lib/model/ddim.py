import numpy as np
import torch
import torch.nn as nn

from lib.utils.ema import EMAHelper
from lib.utils.diffusion import get_beta_schedule, ddpm_steps, generalized_steps


class DDIM:

    def __init__(self,
                 diffusion_config: dict,
                 model_config: dict,
                 device: str):

        """ Creates class instance. """

        self.device = device

        self.model_var_type = model_config['VAR_TYPE']
        self.ema = model_config['EMA']
        self.ema_rate = model_config['EMA_RATE']
        self.sample_type = diffusion_config['SAMPLE_TYPE']
        self.skip_type = diffusion_config['SKIP_TYPE']

        self.betas = get_beta_schedule(
            beta_schedule=diffusion_config['BETA_SCHEDULE'],
            beta_start=diffusion_config['BETA_START'],
            beta_end=diffusion_config['BETA_END'],
            num_diffusion_timesteps=diffusion_config['NUM_DIFFUSION_TIMESTEPS']
        )
        self.betas = torch.from_numpy(self.betas).float().to(device)
        self.num_timesteps = self.betas.shape[0]
        self.timesteps = diffusion_config['NUM_SAMPLE_TIMESTEPS']
        self.eta = diffusion_config['ETA']

        alphas = 1.0 - self.betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = self.betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample_image(self, x, model, last=True, **kwargs):
        """ TODO: Move to GPU fully"""
        if self.sample_type == "generalized":
            if self.skip_type == "uniform":
                skip = self.num_timesteps // self.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError("Skip type not supported")

            x = generalized_steps(x, seq, model, self.betas, eta=self.eta, **kwargs)
        elif self.sample_type == "ddpm_noisy":
            if self.skip_type == "uniform":
                skip = self.num_timesteps // self.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError("Skip type not supported")

            x = ddpm_steps(x, seq, model, self.betas, **kwargs)
        else:
            raise NotImplementedError("Sample type not supported")
        if last:
            x = x[0][-1]
        return x

    def sample_fid(self, model):
        raise NotImplementedError

    def sample_interpolation(self, model: nn.Module):
        raise NotImplementedError

    def sample_sequence(self, model: nn.Module):
        raise NotImplementedError

    def sample(self, model: nn.Module, procedure: str = 'sequence'):
        """
        if getattr(self.config.sampling, "ckpt_id", None) is None:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"), map_location='cpu')
        else:
            states = torch.load(os.path.join(self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"),
                                map_location='cpu')
        """
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        # model.load_state_dict(states[0], strict=True)

        if self.ema:
            ema_helper = EMAHelper(mu=self.ema_rate)
            ema_helper.register(model)
            # ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()

        if procedure == 'fid':
            self.sample_fid(model)
        elif procedure == 'interpolation':
            self.sample_interpolation(model)
        elif procedure == 'sequence':
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedure not defined")
