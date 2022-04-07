import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import torch.nn as nn
import librosa
import torch
import numpy as np
from IPython.display import Audio, display


def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def size_conv(input_size,
              stride: tuple = (0, 0),
              kernel_size: tuple = (0, 0),
              padding: tuple = (0, 0),
              dilatation: int = 1):
    """
        Function returning the size of each layer when Conv2d is applied.

        :param input_size: Initial Size
        :type input_size: tuple
        :param stride: Stride input
        :type stride: tuple
        :param kernel_size: Kernel Size
        :type kernel_size: tuple
        :param padding: Padding Size
        :type padding: tuple
        :param dilatation:
        :type dilatation: int
        :return: Size of the convtranspose output in the given dimension.
        """

    h_out = ((input_size[0] + (2 * padding[0]) - dilatation * (kernel_size[0] - 1) - 1) / stride[0]) + 1
    w_out = ((input_size[1] + (2 * padding[1]) - dilatation * (kernel_size[1] - 1) - 1) / stride[1]) + 1
    return int(h_out), int(w_out)


def size_convtranspose(input_size,
                       stride: tuple = (0, 0),
                       kernel_size: tuple = (0, 0),
                       padding: tuple = (0, 0),
                       dilatation: int = 1,
                       output_padding: int = 0):

    """
    Function returning the size of each layer when Conv2dtransposed is applied.

    :param input_size: Height input
    :type input_size: tuple
    :param stride: Stride input
    :type stride: tuple
    :param kernel_size: Kernel Size
    :type kernel_size: tuple
    :param padding: Padding Size
    :type padding: tuple
    :param dilatation:
    :type dilatation: int
    :param output_padding:
    :type output_padding: int
    :return: Size of the convtranspose output in the given dimension.
    """
    #TODO: It is necessary to do a funtion with both, height and width dimensions and try to reduce the inputs.

    h_out = (input_size[0] - 1) * stride[0] - (2*padding[0]) + dilatation * (kernel_size[0] - 1) + output_padding + 1
    w_out = (input_size[1] - 1) * stride[1] - (2*padding[1]) + dilatation * (kernel_size[1] - 1) + output_padding + 1
    return h_out, w_out


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)


def display_images(model_outputs, epoch):

    """
    Plots the original image and reconstructed image by the machine learning model. This function receives
    a list with a length of 3 like [epochs, original images, reconstructed images].

    Original images and reconstructed images have a shape of [num_images, num_channels, rows, columns].

    :param model_outputs: Outputs of the machine learning model.
    :param epoch: Desire epoch to visualize.
    :type epoch: int
    :return: matplotlib plot
    """

    recon = model_outputs[epoch][2].to("cpu").detach().numpy()
    img = model_outputs[epoch][1].to("cpu").detach().numpy()
    for index in range(recon.shape[0]):
        ax1 = plt.subplot(2, recon.shape[0], index+1)
        ax2 = plt.subplot(2, recon.shape[0], index+recon.shape[0]+1)

        librosa.display.specshow(librosa.power_to_db(recon[index, 0], ref=np.max),
                                 y_axis='mel', fmax=8000,
                                 x_axis='time', ax=ax1)

        librosa.display.specshow(librosa.power_to_db(img[index, 0], ref=np.max),
                                 y_axis='mel', fmax=8000,
                                 x_axis='time', ax=ax2)
    plt.show()


def VAE_loss_function(y_hat, y, mu, logvar):
    """

    :param y_hat:
    :param y:
    :param mu:
    :param logvar:
    :return:
    """
    BCE = nn.functional.binary_cross_entropy(
        y_hat, y.view(-1,784), reduction='sum'
    )
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE + KLD


def train_model_ae(dataloader, model, num_epochs: int, num_images: int, learning_rate: float, device):

    """
    This function trains an autoencoder based on the ConvAE contained in Models.py

    :param dataloader: Torch dataloader for image loading.
    :type dataloader: torch
    :param model: Deep learning model.
    :param num_epochs: Number of epochs. Each epoch pass all the dataset to the model.
    :type num_epochs: int
    :param num_images: Number of images to use of the dataset.
    :type num_images: int
    :param learning_rate: This parameter is used in the optimization pass.
    :type learning_rate: float
    :param device:
    :return:
    """

    model_cuda = model.to(device)
    model_cuda.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_cuda.parameters(), lr=learning_rate, weight_decay=1e-6)
    #step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    outputs = []

    for epoch in range(num_epochs):
        i = 0
        for (img, _, _, _) in dataloader:
            img = img.reshape(-1, img.shape[2], img.shape[3])
            img = torch.unsqueeze(img, 1)

            if i >= num_images:
                break

            #print(f"image: {i}")

            img_cuda = img.to(device)
            # print(f"img shape:{img_cuda.shape}")
            recon = model_cuda(img_cuda)
            # recon_cuda = recon.to(device)
            loss = criterion(recon, img_cuda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #step_lr_scheduler.step()
            i += 1
            # print(f"partial loss: {loss.item():.4f}")

        recon = recon.to('cpu')
        print(f'Epoch:{epoch + 1}, Loss: {loss.item():.4f}')
        outputs.append((epoch, img, recon))
        display_images(outputs, epoch)

