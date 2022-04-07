from inspect import isclass
#import Models
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
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




