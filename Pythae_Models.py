import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 4,
                                 kernel_size=(3, 3),
                                 stride=(2, 2), padding=(0, 0))

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 4,
                                 out_channels=num_hiddens,
                                 kernel_size=(3, 3),
                                 stride=(2, 2), padding=(0, 0))

        self.pooling = nn.MaxPool2d(2)

    def forward(self, inputs):

        # print("Working with new encoder")
        print(f"inputs:{inputs.shape}")
        x = self._conv_1(inputs)
        x = F.leaky_relu(x)

        x = self._conv_2(x)
        x = F.leaky_relu(x)

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens):
        super(Decoder, self).__init__()

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 4,
                                                kernel_size=(3, 3),
                                                stride=(2, 2), padding=(0, 0),  output_padding=(0, 0))

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 4,
                                                out_channels=1,
                                                kernel_size=(3, 3),
                                                stride=(2, 2), padding=(0, 0), output_padding=(0, 0))

        self.dropout = nn.Dropout2d(p=0.2)

        self.transpooling = nn.MaxUnpool2d(2)

        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def forward(self, inputs):

        #print("Working with new decoder")

        x = self._conv_trans_1(inputs)
        x = F.leaky_relu(x)

        x = self._conv_trans_2(x)
        # x = F.leaky_relu(x)
        # print(f"convtr6: {x.shape}")

        x = self.Sigmoid(x)

        return x