import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class ClusteringLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)


class VAE(nn.Module):
    def __init__(self, in_channels=1, num_clusters=31, num_hiddens=64, zdim=128, leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(VAE, self).__init__()
        self.activations = activations
        # bias = True
        self.zdim = zdim
        self.pretrained = False
        self.num_clusters = num_clusters
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.num_hiddens // 4,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=bias)

        self.conv2 = nn.Conv2d(in_channels=self.num_hiddens // 4,
                               out_channels=self.num_hiddens,
                               kernel_size=4,
                               stride=2,
                               padding=0,
                               bias=bias)

        self.conv3 = nn.Conv2d(in_channels=self.num_hiddens,
                               out_channels=self.zdim,
                               kernel_size=4,
                               stride=2,
                               padding=0,
                               bias=bias)

        # lin_features_len = ((input_shape[0]//2//2-1) // 2) * ((input_shape[0]//2//2-1) // 2) * self.num_hiddens
        # self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        # self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        self.deconv3 = nn.ConvTranspose2d(self.zdim, self.num_hiddens,
                                          kernel_size=4,
                                          stride=2,
                                          padding=0,
                                          output_padding=0,
                                          bias=bias)
        self.deconv2 = nn.ConvTranspose2d(self.num_hiddens,
                                          self.num_hiddens // 4,
                                          kernel_size=4,
                                          stride=2,
                                          padding=0,
                                          output_padding=0,
                                          bias=bias)
        self.deconv1 = nn.ConvTranspose2d(self.num_hiddens // 4, self.in_channels,
                                          kernel_size=4,
                                          stride=1,
                                          padding=0,
                                          output_padding=0,
                                          bias=bias)

        self.clustering = ClusteringLayer(num_clusters, num_clusters)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss


        print(f"inputs shape {x.shape}")
        x = self.relu(self.conv1(x))
        print(f"conv1: {x.shape}")
        x = self.relu(self.conv2(x))
        print(f"conv2: {x.shape}")
        x = self.relu(self.conv3(x))
        print(f"conv3: {x.shape}")
        self.featureDim = x.shape[1] * x.shape[2] * x.shape[3]
        self.encFC1 = nn.Linear(self.featureDim, self.zDim)
        self.encFC2 = nn.Linear(self.featureDim, self.zDim)
        x = x.view(-1, self.featureDim)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logvar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        self.decFC1 = nn.Linear(self.zDim, self.featureDim)
        x = self.relu(self.decFC1(z))
        print(f"decFC1: {x.shape}")
        x = x.view(-1, self.num_hiddens, self.xshape[2], self.xshape[3])
        print(f"view: {x.shape}")
        x = self.relu(self.decConv1(x))
        print(f"decC1: {x.shape}")
        x = self.relu(self.decConv2(x))
        print(f"decC2: {x.shape}")
        # x = F.relu(self.decConv3(x))
        x = self.sig(self.decConv4(x))
        print(f"deconv2: {x.shape}")
        #x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):
        mu, logVar = self.encoder(x)
        clustering_out = self.clustering(mu)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)

        return out, clustering_out, z
