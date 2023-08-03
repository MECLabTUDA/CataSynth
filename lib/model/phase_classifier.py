""" Following https://www.synapse.org/#!Synapse:syn22394101/wiki/605469 """

import torch
import torch.nn as nn

from torchvision.models import densenet169, DenseNet169_Weights


class PhaseClassifier(nn.Module):

    def __init__(self,
                 img_size: tuple = (224, 224),
                 num_phase_labels: int = 19,
                 num_tool_labels: int = 21,
                 fc_ch: tuple = (32, 32),
                 dropout_p: float = 0.3):

        super(PhaseClassifier, self).__init__()

        self.enc = densenet169(weights=DenseNet169_Weights.DEFAULT)
        # self.enc = self.enc.features
        fake_input = torch.rand(size=(1, 3, *img_size))
        with torch.no_grad():
            fake_ft = self.enc(fake_input)
            num_ft = torch.flatten(fake_ft, start_dim=0).shape[0]
        print(num_ft)

        self.stacked_rnn = nn.LSTM(input_size=num_ft, hidden_size=fc_ch[0], num_layers=2)

        self.trans = nn.Linear(in_features=num_ft, out_features=fc_ch[0])

        self.phase_fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=fc_ch[0], out_features=fc_ch[0]),
            nn.BatchNorm1d(fc_ch[0]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=fc_ch[0], out_features=fc_ch[1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=fc_ch[1], out_features=num_phase_labels)
        )

        self.tool_fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=fc_ch[0], out_features=fc_ch[0]),
            nn.BatchNorm1d(fc_ch[0]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=fc_ch[0], out_features=fc_ch[1]),
            nn.BatchNorm1d(fc_ch[1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=fc_ch[1], out_features=num_tool_labels)
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        ft = self.enc(x)

        rnn_ft = self.stacked_rnn(ft)

        h_p_t = self.phase_fc(rnn_ft)

        h_t = self.phase_fc(ft)

        q_p_t = self.tool_fc(rnn_ft)

        q_t = self.tool_fc(ft)

        return h_t, h_p_t, q_t, q_p_t

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



class FullyConvClassifier(nn.Module):

    def __init__(self,
                 img_size: tuple = (128, 128),
                 num_phase_labels: int = 19,
                 num_tool_labels: int = 21,
                 fc_ch: tuple = (32, 128),
                 dropout_p: float = 0.3
                 ):

        super(FullyConvClassifier, self).__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=fc_ch[0], kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(fc_ch[0]),

            nn.Conv2d(in_channels=fc_ch[0], out_channels=fc_ch[0]*2, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(fc_ch[0] * 2),

            nn.Conv2d(in_channels=fc_ch[0]*2, out_channels=fc_ch[1], kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(fc_ch[1]),

            nn.Conv2d(in_channels=fc_ch[1], out_channels=fc_ch[1]*2, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(fc_ch[1]*2),
        )

        fake_input = torch.rand(size=(1, 3, *img_size))
        with torch.no_grad():
            fake_ft = self.enc(fake_input)
            num_ft = tuple(torch.flatten(fake_ft, start_dim=0).shape)[0]
        print(num_ft)

        self.phase_fc1 = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features=num_ft, out_features=fc_ch[-1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(fc_ch[-1]),

            nn.Dropout(dropout_p),
            nn.Linear(in_features=fc_ch[-1], out_features=fc_ch[-2]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(fc_ch[-2]),

            nn.Dropout(dropout_p),
            nn.Linear(in_features=fc_ch[-2], out_features=num_phase_labels)
        )
        self.tool_fc1 = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features=num_ft, out_features=fc_ch[-1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(fc_ch[-1]),

            nn.Dropout(dropout_p),
            nn.Linear(in_features=fc_ch[-1], out_features=fc_ch[-2]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(fc_ch[-2]),

            nn.Dropout(dropout_p),
            nn.Linear(in_features=fc_ch[-2], out_features=num_tool_labels)
        )



