import torch
import torch.nn as nn
from lib.model.layers import nonlinearity, get_timestep_embedding

from .unet import UNet


class ConditionalUNet(UNet):

    def __init__(self,
                 data_config: dict,
                 model_config: dict,
                 diffusion_config: dict,
                 num_phase_labels: int = None,
                 num_semantic_labels: int = None):

        super(ConditionalUNet, self).__init__(data_config, model_config, diffusion_config)

        if num_phase_labels is not None:
            self.phase_label_embedding = nn.Embedding(num_embeddings=num_phase_labels, embedding_dim=self.temb_ch)
        if num_semantic_labels is not None:
            self.semantic_label_embedding = nn.ModuleList([
                torch.nn.Linear(num_semantic_labels,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

    def forward(self, x, t, mask=None, phase_y=None, tool_y=None):

        assert x.shape[2] == x.shape[3] == self.data_shape[-1]

        # Timestep embedding
        time_emb = get_timestep_embedding(t, self.ch)
        time_emb = self.temb.dense[0](time_emb)
        time_emb = nonlinearity(time_emb)
        time_emb = self.temb.dense[1](time_emb)

        # Append label embedding to timestep embedding
        if phase_y is not None:
            # Reshaping phase label embedding to one dim
            time_emb += self.phase_label_embedding(phase_y.squeeze(-1))
        if tool_y is not None:
            semantic_emb = self.semantic_label_embedding[0](tool_y)
            nonlinearity(semantic_emb)
            semantic_emb = self.semantic_label_embedding[1](semantic_emb)
            time_emb += semantic_emb

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], time_emb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Bottom layers
        h = hs[-1]
        h = self.mid.block_1(h, time_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, time_emb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), time_emb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
