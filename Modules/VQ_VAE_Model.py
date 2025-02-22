class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 4,
                                 kernel_size=(3, 3),
                                 stride=(2, 2), padding=(0, 0))
        # self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 4,
        #                          out_channels=num_hiddens // 2,
        #                          kernel_size=(8, 8),
        #                          stride=(2, 2), padding=(0, 0))

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens // 4,
                                 out_channels=num_hiddens,
                                 kernel_size=(3, 3),
                                 stride=(2, 2), padding=(0, 0))

        # self._conv_4 = nn.Conv2d(in_channels=num_hiddens // 2,
        #                          out_channels=num_hiddens,
        #                          kernel_size=(4, 4),
        #                          stride=(2, 2), padding=(0, 0))
        # self._residual_stack = ResidualStack(in_channels=num_hiddens,
        #                                      num_hiddens=num_hiddens,
        #                                      num_residual_layers=num_residual_layers,
        #                                      num_residual_hiddens=num_residual_hiddens)

        self.pooling = nn.MaxPool2d(2)

    def forward(self, inputs):

        # print("Working with new encoder")
        print(f"inputs:{inputs.shape}")
        x = self._conv_1(inputs)
        x = F.leaky_relu(x)
        # print(f"conv1: {x.shape}")

        # x = self.pooling(x)
        # print(f"pooling1: {x.shape}")

        # x = self._conv_2(x)
        # x = F.leaky_relu(x)
        # # print(f"conv2: {x.shape}")

        # x = self.pooling(x)
        # print(f"pooling2: {x.shape}")

        x = self._conv_3(x)
        x = F.leaky_relu(x)
        # print(f"conv3: {x.shape}")

        # x = self._conv_4(x)
        # x = F.relu(x)
        # # print(f"conv4: {x.shape}")

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens):
        super(Decoder, self).__init__()

        # self._conv_1 = nn.Conv2d(in_channels=embedding_dim,
        #                          out_channels=num_hiddens,
        #                          kernel_size=(4, 4),
        #                          stride=(2, 2), padding=(0, 0))

        # self._residual_stack = ResidualStack(in_channels=num_hiddens,
        #                                      num_hiddens=num_hiddens,
        #                                      num_residual_layers=num_residual_layers,
        #                                      num_residual_hiddens=num_residual_hiddens)

        # self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
        #                                         out_channels=embedding_dim,
        #                                         kernel_size=(4, 4),
        #                                         stride=(2, 2), padding=(0, 0))

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=embedding_dim,
                                                out_channels=num_hiddens,
                                                kernel_size=(4, 4),
                                                stride=(2, 2), padding=(0, 0), output_padding=(0, 0))

        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 4,
                                                kernel_size=(3, 3),
                                                stride=(2, 2), padding=(0, 0),  output_padding=(0, 0))

        # self._conv_trans_4 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
        #                                         out_channels=num_hiddens // 4,
        #                                         kernel_size=(4, 4),
        #                                         stride=(2, 2), padding=(0, 0), output_padding=(0, 0))

        # self._conv_trans_5 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
        #                                         out_channels=num_hiddens // 4,
        #                                         kernel_size=(8, 8),
        #                                         stride=(2, 2), padding=(0, 0), output_padding=(0, 0))

        self._conv_trans_6 = nn.ConvTranspose2d(in_channels=num_hiddens // 4,
                                                out_channels=1,
                                                kernel_size=(3, 3),
                                                stride=(2, 2), padding=(0, 0), output_padding=(0, 0))

        self.dropout = nn.Dropout2d(p=0.2)

        self.transpooling = nn.MaxUnpool2d(2)

        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def forward(self, inputs):

        #print("Working with new decoder")
        # print(f"inputs: {inputs.shape}")

        # x = self._conv_1(inputs)
        # x = F.relu(x)
        # print(f"conv1: {x.shape}")

        # x = self._conv_trans_1(x)
        # x = F.relu(x)
        # print(f"convtr1: {x.shape}")

        x = self._conv_trans_2(inputs)
        x = F.leaky_relu(x)
        # print(f"convtr2: {x.shape}")

        # x = self.transpooling(x)
        # print(f"transpooling: {x.shape}")

        x = self._conv_trans_3(x)
        x = F.leaky_relu(x)
        # print(f"convtr3: {x.shape}")

        # x = self.transpooling(x)
        # print(f"transpooling: {x.shape}")

        # x = self._conv_trans_4(x)
        # x = F.relu(x)
        # # print(f"convtr4: {x.shape}")

        # x = self._conv_trans_5(x)
        # x = F.relu(x)
        # # print(f"convtr5: {x.shape}")

        x = self._conv_trans_6(x)
        # x = F.leaky_relu(x)
        # print(f"convtr6: {x.shape}")

        x = self.Sigmoid(x)

        return x


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Model(nn.Module):
    def __init__(self, num_hiddens, num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._encoder = Encoder(1, num_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      padding=0,
                                      kernel_size=(4, 4),
                                      stride=(2, 2))
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        # print(f"Z shape: {z.shape}")

        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity