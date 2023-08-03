import torch
import torch.nn as nn


def decay_gauss_std(net):
    for m in net.modules():
        if isinstance(m, GaussianNoise):
            m.decay_step()


class GaussianNoise(nn.Module):
    def __init__(self,
                 std: float = .1,
                 decay_rate: float = 0.):

        super(GaussianNoise, self).__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x


class Generator(nn.Module):

    def __init__(self,
                 label_embedding_dim: int,
                 latent_dim: int,  # Dim. of noise / z
                 base_hidden_dim: int = 32,
                 dropout: float = 0.0,
                 n_phase_classes: int = None,
                 n_tool_dims: int = None,
                 ):
        super(Generator, self).__init__()

        self.latent_map_dim = 1024
        gen_in_ch = 1024
        self.cond_out_ch = 64

        if n_phase_classes is not None:
            self.phase_label_condition_generator = nn.Sequential(
                nn.Embedding(n_phase_classes, label_embedding_dim),
                nn.Linear(label_embedding_dim, label_embedding_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(label_embedding_dim, self.cond_out_ch * 4 * 4),
                nn.LeakyReLU(0.2, inplace=True),
            )
            gen_in_ch += self.cond_out_ch

        if n_tool_dims is not None:
            self.tool_label_condition_generator = nn.Sequential(
                nn.Linear(n_tool_dims, label_embedding_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(label_embedding_dim, self.cond_out_ch * 4 * 4),
                nn.LeakyReLU(0.2, inplace=True)
            )
            gen_in_ch += self.cond_out_ch

        self.latent = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * self.latent_map_dim),
            nn.BatchNorm1d(4 * 4 * self.latent_map_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(gen_in_ch, base_hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_hidden_dim * 8, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),

            nn.ConvTranspose2d(base_hidden_dim * 8, base_hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_hidden_dim * 4, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),

            nn.ConvTranspose2d(base_hidden_dim * 4, base_hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_hidden_dim * 2, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),

            nn.ConvTranspose2d(base_hidden_dim * 2, base_hidden_dim * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_hidden_dim * 1, momentum=0.1, eps=0.8),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_hidden_dim * 1, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self,
                noise_vector: torch.Tensor,
                phase_label: torch.Tensor = None,
                tool_label: torch.Tensor = None) -> torch.Tensor:

        inputs = []

        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, self.latent_map_dim, 4, 4)
        inputs.append(latent_output)

        if phase_label is not None:
            phase_label_output = self.phase_label_condition_generator(phase_label)
            phase_label_output = phase_label_output.view(-1, self.cond_out_ch, 4, 4)
            inputs.append(phase_label_output)

        if tool_label is not None:
            tool_label_output = self.tool_label_condition_generator(tool_label)
            tool_label_output = tool_label_output.view(-1, self.cond_out_ch, 4, 4)
            inputs.append(tool_label_output)

        concat = torch.cat(inputs, dim=1)

        image = self.model(concat)

        return image


class Discriminator(nn.Module):

    def __init__(self,
                 img_shape: tuple = (3, 128, 128),
                 label_embedding_dim: int = 100,
                 base_hidden_dim: int = 64,
                 dropout: float = 0.0,
                 noise_std: float = 0.,
                 noise_decay_rate: float = 0.,
                 n_phase_classes: int = None,
                 n_tool_dims: int = None,
                 ):
        super(Discriminator, self).__init__()

        print("##### n_phase_classes: ", n_phase_classes)

        self.img_shape = img_shape
        disc_in_ch = img_shape[0]

        if n_phase_classes is not None:
            self.phase_label_condition_disc = nn.Sequential(
                nn.Embedding(n_phase_classes, label_embedding_dim),
                nn.Linear(label_embedding_dim, img_shape[0] * img_shape[1] * img_shape[2]))
            disc_in_ch += 3

        if n_tool_dims is not None:
            self.tool_label_condition_disc = nn.Sequential(
                nn.Linear(n_tool_dims, label_embedding_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(label_embedding_dim, img_shape[0] * img_shape[1] * img_shape[2]),
                nn.LeakyReLU(0.2, inplace=True)
            )
            disc_in_ch += 3

        self.model = nn.Sequential(
            GaussianNoise(noise_std, noise_decay_rate) if noise_std > 0. else nn.Identity(),
            nn.Conv2d(disc_in_ch, base_hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(noise_std, noise_decay_rate) if noise_std > 0. else nn.Identity(),
            nn.Conv2d(base_hidden_dim, base_hidden_dim * 2, 4, 3, 2, bias=False),
            nn.BatchNorm2d(base_hidden_dim * 2, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(noise_std, noise_decay_rate) if noise_std > 0. else nn.Identity(),
            nn.Conv2d(base_hidden_dim * 2, base_hidden_dim * 4, 4, 3, 2, bias=False),
            nn.BatchNorm2d(base_hidden_dim * 4, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(noise_std, noise_decay_rate) if noise_std > 0. else nn.Identity(),
            nn.Conv2d(base_hidden_dim * 4, base_hidden_dim * 8, 4, 3, 2, bias=False),
            nn.BatchNorm2d(base_hidden_dim * 8, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            GaussianNoise(noise_std, noise_decay_rate) if noise_std > 0. else nn.Identity(),
            nn.Conv2d(base_hidden_dim * 8, base_hidden_dim * 16, 4, 3, 2, bias=False),
            nn.BatchNorm2d(base_hidden_dim * 16, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(start_dim=1),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            GaussianNoise(noise_std, noise_decay_rate) if noise_std > 0. else nn.Identity(),
            nn.LazyLinear(1),
            # nn.Sigmoid()
        )

    def forward(self,
                img: torch.Tensor,
                phase_label: torch.Tensor = None,
                tool_label: torch.Tensor = None) -> torch.Tensor:

        input = [img]

        if phase_label is not None:
            phase_label_output = self.phase_label_condition_disc(phase_label)
            phase_label_output = phase_label_output.view(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
            input.append(phase_label_output)

        if tool_label is not None:
            tool_label_output = self.tool_label_condition_disc(tool_label)
            tool_label_output = tool_label_output.view(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
            input.append(tool_label_output)

        concat = torch.cat(input, dim=1)
        output = self.model(concat)

        return output
