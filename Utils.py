from inspect import isclass
#import Models
import matplotlib.pyplot as plt
import librosa
import torch
from torchvision.utils import make_grid
from IPython.display import display, Audio


def size_conv(input_size,
              stride: tuple = (0, 0),
              kernel_size: tuple = (0, 0),
              padding: tuple = (0, 0),
              dilatation: int = 1):
    """
        Function returning the size of each layer when Conv2d is applied.

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


def play_audio(waveform, sample_rate):

    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio((waveform[0]), rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def plot_spectrogram(spec, title=None, ylabel: str = 'freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    #axs.set_title(title or 'Spectrogram (db)')
    #axs.set_ylabel(ylabel)
    #axs.set_xlabel('frame')
    axs.set_axis_off()
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    #fig.colorbar(im, ax=axs)
    fig.savefig("try2")
    plt.show(block=False)
    fig.savefig("try2", bbox_inches='tight',transparent=True, pad_inches=0.0)


def test_vq_vae(model, iterator):
    model.eval()
    (valid_originals, _,_) = next( iterator)
    valid_originals = torch.reshape(valid_originals, (valid_originals.shape[0] * valid_originals.shape[1],
                                                  valid_originals.shape[2], valid_originals.shape[3]))
    valid_originals = torch.unsqueeze(valid_originals,1)

    valid_originals = valid_originals.to(device)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)
    output = torch.cat((valid_originals[0:8], valid_reconstructions[0:8]), 0)
    img_grid = make_grid(output, nrow=8, pad_value=20)

    recon_error = F.mse_loss(valid_originals, valid_reconstructions)

    fig, ax = plt.subplots(figsize=(20,5))
    ax.imshow(img_grid[1, :, :].cpu(), vmin=0, vmax=1)
    ax.axis("off")
    plt.show()
    return fig, recon_error

