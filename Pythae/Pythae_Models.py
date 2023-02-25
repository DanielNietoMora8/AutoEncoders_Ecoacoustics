import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pythae.models.nn import BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder
from pythae.models.base.base_utils import ModelOutput


class My_encoder(BaseEncoder):

    def __init__(self, num_hiddens: int=64):
        BaseEncoder.__init__(self)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_hiddens // 8, kernel_size=8, stride=3, padding=0),  # N, 256, 127, 8004
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 8, num_hiddens // 4, kernel_size=8, stride=3, padding=0),  # N, 512, 125,969
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 4, num_hiddens // 2, kernel_size=4, stride=3, padding=0),  # N, 512, 125,969
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=2, stride=2, padding=0),  # N, 512, 125,969
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        embedding = self.encoder(x)
        output = ModelOutput(
            embedding=embedding,
            # log_covariance=log_var # for VAE based models
        )
        return output


class My_decoder(BaseDecoder):

    def __init__(self, in_channels=1, num_hiddens: int = 64):
        BaseDecoder.__init__(self)
        self.decoder = nn.Sequential(  # This is like go in opposite direction respect the encoder
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens // 2, num_hiddens // 4, kernel_size=4, stride=3, padding=0,
                               output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens // 4, num_hiddens // 8, kernel_size=8, stride=3, padding=0,
                               output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens // 8, 1, kernel_size=8, stride=3, padding=0, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor):
           reconstruction = self.decoder(z)
           output = ModelOutput(
                reconstruction=reconstruction
            )
           return output

