import torch.nn as nn
from enum import Enum
import sys

class LinearAE(nn.Module):

    """
    Linear layer autoencoder made to reconstruct 1d signals corresponding to ecoacustics audio.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, d),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, 28 * 28),
            nn.Tanh()
        )

    def forward(self, y):
        h = self.encoder(y)
        y_hat = self.decoder(h)
        return y_hat

    def forward(self, x):
        """
        Method to compute a signal output based on the performed model.

        :param x: Input signal as a tensors.
        :type x: torch.tensor
        :return: Reconstructed signal
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvAE(nn.Module):

    """
    Convolutional autoencoder made to reconstruct the audios spectrograms generated by the EcoDataTesis dataloader.
    """

    def __init__(self):
        """
        Constructor of the convolutional autoencoder model.
        """
        super().__init__()
        # TODO: To design the final architechture considering the spectrograms sizes.
        # TODO: To correct the current sizes of the decoder.

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (2, 32), stride=(2, 4), padding=0),  # N, 256, 127, 8004
            nn.ReLU(),
            nn.Conv2d(8, 16, (2, 32), stride=(2, 4), padding=0),  # N, 256, 127, 8004
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 32), stride=(2, 4), padding=0),  # N, 512, 125,969
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 32), stride=(2, 4), padding=0),  # N, 512, 125,969
        )
        self.decoder = nn.Sequential(  # This is like go in opposite direction respect the encoder
            nn.ConvTranspose2d(64, 32, (2, 32), stride=(2, 4), padding=0, output_padding=(0, 2)),  # N, 32, 126,8000
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2, 32), stride=(2, 4), padding=0, output_padding=(0, 2)),  # N, 32, 126,8000
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (2, 32), stride=(2, 4), padding=0, output_padding=(0, 0)),  # N, 32, 127,64248
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, (2, 32), stride=(2, 4), padding=0, output_padding=(1, 3)),  # N, 32, 127,64248
            nn.ReLU(),
            nn.Sigmoid()

        )

    def forward(self, x):
        """
        Method to compute an image output based on the performed architecture.

        :param x: Input spectrogram images as tensors.
        :type x: torch.tensor
        :return: Reconstructed images
        """
        print("---------------------------------------------------")
        print(f"x_shape:{x.shape}")
        encoded = self.encoder(x)
        print(encoded.shape)
        decoded = self.decoder(encoded)
        print(decoded.shape)
        return decoded


class LinearVAE(nn.Module):
    """
    A variational autoencoder that uses a linear architecture
    """

    def __init__(self):
        """
        Constructor of the Conv Variational autoencoder model.
        """
        super(LinearVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, d**2),
            nn.ReLU(),
            nn.Linear(d**2, d*2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, d**2),
            nn.ReLU(),
            nn.Linear(d**2, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """
        This method computes the variational part of the model. Here it is sampled an instance from the distribution..

        :param mu: Median of the distribution.
        :type mu: torch tensor
        :param logvar: logarithm of the deviation.
        :type logvar: torch tensor
        :return: z
        """

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, y):
        """
       This is the forward pass.

       :param y: Spectrogram as a tensor.
       :type y: torch tensor
       :return: Reconstructed image
       :type: torch tensor
       """
        mu_logvar = self.encoder(y.view(-1, 784)).view(-1, 2, d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class CnnVAE(nn.Module):

    """
    A Convolutional Variational Autoencoder
    """
    def __init__(self, imgChannels=1, featureDim=32*20*20, zDim=256):
        """
        Constructor of the Conv Variational autoencoder model.
        """
        super(CnnVAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):
        """
        Encoding part

        :param x: Input image
        :type x: torch tensor
        """

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        print(f"previously redim: {x.shape}")
        x = x.view(-1, 32*20*20)
        print(f"encoder redim: {x.shape}")
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar


    def reparameterize(self, mu, logVar):
        """
        This method computes the variational part of the model. Here it is sampled an instance from the distribution..

        :param mu: Median of the distribution.
        :type mu: torch tensor
        :param logvar: logarithm of the deviation.
        :type logvar: torch tensor
        :return: z
        """
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        """
        Encoding part

        :param z: Embedding
        :type z: torch tensor
        """
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 20, 20)
        print(f"redim = {x.shape}")
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):
        """
        Forward pass
        """
        print(f'x shape is: {x.shape}')
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar


class ConvAE2(nn.Module):

    """
    Convolutional autoencoder made to reconstruct the audios spectrograms generated by the EcoDataTesis dataloader.
    """

    def __init__(self):
        """
        Constructor of the convolutional autoencoder model.
        """
        super().__init__()
        # TODO: To design the final architechture considering the spectrograms sizes.
        # TODO: To correct the current sizes of the decoder.

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (8, 32), stride=(2, 2), padding=1),  # N, 256, 127, 8004
            nn.ReLU(),
            nn.Conv2d(8, 16, (4, 16), stride=(2, 2), padding=1),  # N, 256, 127, 8004
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 4), stride=(2, 2), padding=1),  # N, 512, 125,969
            # nn.ReLU(),
            # nn.Conv2d(128, 164, (8, 2), stride=(1, 1), padding=0),  # N, 512, 125,969
            # nn.ReLU(),
            # nn.Conv2d(164, 228, (3, 1), stride=(1, 1), padding=0),  # N, 512, 125,969
            # nn.ReLU(),
            # nn.Conv2d(128, 160, (29, 4), stride=(1, 1), padding=0),  # N, 512, 125,969
            # nn.ReLU(),
            # nn.Conv2d(160, 200, (2, 1), stride=(1, 1), padding=0),  # N, 512, 125,969
        )
        self.decoder = nn.Sequential(  # This is like go in opposite direction respect to the encoder
            # nn.ConvTranspose2d(200, 160, (2, 1), stride=(1, 1), padding=0, output_padding=(0, 1)),  # N, 32, 126,8000
            # nn.ReLU(),
            # nn.ConvTranspose2d(160, 128, (29, 4), stride=(1, 1), padding=0, output_padding=(0, 0)),  # N, 32, 126,8000
            # nn.ReLU(),
            # nn.ConvTranspose2d(228, 164, (3, 1), stride=(1, 1), padding=0, output_padding=(0, 0)),  # N, 32, 126,8000
            # nn.ReLU(),
            # nn.ConvTranspose2d(164, 128, (8, 2), stride=(1, 1), padding=0, output_padding=(0, 0)),  # N, 32, 126,8000
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2, 4), stride=(2, 2), padding=1, output_padding=(1, 0)),  # N, 32, 126,8000
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (4, 16), stride=(2, 2), padding=1, output_padding=(0, 1)),  # N, 32, 127,64248
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, (8, 32), stride=(2, 2), padding=1, output_padding=(1, 1)),  # N, 32, 127,64248
            nn.ReLU(),
            nn.Sigmoid()

        )

    def forward(self, x):

        """
        Method to compute an image output based on the performed model.

        :param x: Input spectrogram images as tensors.
        :type x: torch.tensor
        :return: Reconstructed images
        """
        # print("-------------working-------------------------")
        #print(f"x_shape:{x.shape}")
        encoded = self.encoder(x)
        #print(f"encoded_shape:{encoded.shape}")
        #print(f"Bytes after encoding: {sys.getsizeof(encoded.storage())}")
        decoded = self.decoder(encoded)
        #print(f"decoder_shape: {decoded.shape}")
        return decoded

class ConvAE3(nn.Module):

    """
    Convolutional autoencoder made to reconstruct the audios spectrograms generated by the EcoDataTesis dataloader.
    """

    def __init__(self):
        """
        Constructor of the convolutional autoencoder model.
        """
        super().__init__()
        # TODO: To design the final architechture considering the spectrograms sizes.
        # TODO: To correct the current sizes of the decoder.

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (8, 32), stride=(2, 2), padding=1),  # N, 256, 127, 8004
            nn.ReLU(),
            nn.Conv2d(8, 16, (4, 16), stride=(2, 2), padding=1),  # N, 256, 127, 8004
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 4), stride=(2, 2), padding=1),  # N, 512, 125,969
            # nn.ReLU(),
            # nn.Conv2d(128, 164, (8, 2), stride=(1, 1), padding=0),  # N, 512, 125,969
            # nn.ReLU(),
            # nn.Conv2d(164, 228, (3, 1), stride=(1, 1), padding=0),  # N, 512, 125,969
            # nn.ReLU(),
            # nn.Conv2d(128, 160, (29, 4), stride=(1, 1), padding=0),  # N, 512, 125,969
            # nn.ReLU(),
            # nn.Conv2d(160, 200, (2, 1), stride=(1, 1), padding=0),  # N, 512, 125,969
        )
        self.decoder = nn.Sequential(  # This is like go in opposite direction respect to the encoder
            # nn.ConvTranspose2d(200, 160, (2, 1), stride=(1, 1), padding=0, output_padding=(0, 1)),  # N, 32, 126,8000
            # nn.ReLU(),
            # nn.ConvTranspose2d(160, 128, (29, 4), stride=(1, 1), padding=0, output_padding=(0, 0)),  # N, 32, 126,8000
            # nn.ReLU(),
            # nn.ConvTranspose2d(228, 164, (3, 1), stride=(1, 1), padding=0, output_padding=(0, 0)),  # N, 32, 126,8000
            # nn.ReLU(),
            # nn.ConvTranspose2d(164, 128, (8, 2), stride=(1, 1), padding=0, output_padding=(0, 0)),  # N, 32, 126,8000
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2, 4), stride=(2, 2), padding=1, output_padding=(1, 0)),  # N, 32, 126,8000
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (4, 16), stride=(2, 2), padding=1, output_padding=(0, 1)),  # N, 32, 127,64248
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, (8, 32), stride=(2, 2), padding=1, output_padding=(1, 1)),  # N, 32, 127,64248
            nn.ReLU(),
            nn.Sigmoid()

        )

    def forward(self, x):

        """
        Method to compute an image output based on the performed model.

        :param x: Input spectrogram images as tensors.
        :type x: torch.tensor
        :return: Reconstructed images
        """
        # print("-------------working-------------------------")
        #print(f"x_shape:{x.shape}")
        encoded = self.encoder(x)
        #print(f"encoded_shape:{encoded.shape}")
        #print(f"Bytes after encoding: {sys.getsizeof(encoded.storage())}")
        decoded = self.decoder(encoded)
        #print(f"decoder_shape: {decoded.shape}")
        return decoded


class ModelName(str, Enum):

    """
    ModelName is an enumeration and it contains the model names as enumeration members. For more information
    visit the nex website https://docs.python.org/3/library/enum.html
    """
    LinearAE = "LinearAE"
    ConvAE = "ConvAE"
    LinearVAE = "LinearVAE"
    CnnVAE = "CnnVAE"

