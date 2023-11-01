import torch
import torchaudio
from IPython import get_ipython
import os
import numpy as np
from pathlib import Path
import torchaudio.transforms as F
from torch.utils.data import Dataset
import pandas as pd


class SoundscapeData(Dataset):

    """
    JaguasData is the main class used by the dataloader function allowing access to acoustic data of jaguas,
    and then training and testing a deep learning model. This version allows to define the audio length
    to return.
    """

    def __init__(self, root_path: str, dataframe_path: str, audio_length: int, ext: str = "wav",
                 win_length: int = 255, original_length: int = 60, **kwargs):

        """
        This function is used to initialize the Dataloader, here path and root of files are defined.

        :param root_path: Files directory.
        :type root_path: str
        :param audio_length: Desired audio length partitions.
        :type audio_length: int
        :param ext: Audios format (e.g., .wav).
        :type ext: str
        :param ext: Audios extension (ex: .wav)
        :param win_length: Window length used to compute the fourier transform.
        :type win_length: int
        :param original_length: Duration in seconds of the original audios.
        :type original_length:
        """

        if 'google.colab' in str(get_ipython()):
            dir_root = "/content/drive/Shareddrives/"
        elif "zmqshell" in str(get_ipython()):
            dir_root = "/"
        else:
            dir_root = "G:/Unidades compartidas/"

        self.audio_length = audio_length
        self.original_length = original_length
        self.root_path = dir_root+root_path
        print(self.root_path)
        self.win_length = win_length
        self.files = []
        self.kwargs = kwargs

        df_folders = pd.read_csv(dataframe_path)
        if ("filters" in self.kwargs):
            self.filters = kwargs["filters"]
            for key in self.filters.keys():
                df_folders = df_folders[df_folders[key] == self.filters[key]]
        self.files = list(df_folders["Filename"])

        # for i in range(len(self.folders)):
        #     df_recorder = pd.read_csv(f"Jaguas\Complementary_Files\Audios_Jaguas\{self.folders[i]}.csv")
        #     df_recorder = df_recorder[df_recorder["Intensity_Category"] == "No_rain"]
        #     self.files += list(df_recorder["Filename"])

    def __getitem__(self, index):

        """
        Function used to return audios and its respective spectrogram. Partitions of audios are made based on the
        users' parameterization. An additional axes for partitions is returned.

        :param index: index indicates the number of data to return.
        :returns:
            :spec: Spectrogram of the indexed audios.
            :type spec: torch.tensor
            :record: Array of indexed audios in monophonic format.
            :type record: numpy.array
            :label: Dictionary of labels including recorder, hour, minute and second keys.
            :type label: Dictionary
            :path_index: File directory.
            :type path index: String

        """
        if 'google.colab' in str(get_ipython()) or "zmqshell" in str(get_ipython()):
            delimiter = "/"
        else:
            delimiter = "\\"

        delimiter = "_"
        path_index = self.files[index]
        recorder_str = str(path_index).split(delimiter)[0]
        recorder = int(recorder_str[1:3])
        hour = int(str(path_index).split(delimiter)[2][0:2])
        minute = int(str(path_index).split(delimiter)[2][2:4])
        second = int(str(path_index).split(delimiter)[2][4:6])
        label = {"recorder": np.repeat(recorder, self.original_length//self.audio_length),
                 "hour": np.repeat(hour, self.original_length//self.audio_length),
                 "minute": np.repeat(minute, self.original_length//self.audio_length),
                 "second": np.repeat(second, self.original_length//self.audio_length)}

        audio_path = self.root_path + "/" + recorder_str + "_m" + "/" + path_index
        record, sr = torchaudio.load(audio_path)
        resampling = 44100 // 2
        audio_len = self.audio_length * resampling
        record = torch.mean(record, dim=0, keepdim=True)
        record = torchaudio.transforms.Resample(sr, resampling)(record)
        missing_padding = resampling * self.original_length - record.shape[1]
        padding = torch.zeros([1, missing_padding])
        record = torch.cat((record, padding), axis=1)
        record = record[:, :audio_len * (record.shape[1] // audio_len)]
        record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
        win_length = self.win_length
        nfft = int(np.round(1*win_length))

        if "spectrogram_type" in self.kwargs and self.kwargs["spectrogram_type"] == "Mel":
            spec = torchaudio.transforms.MelSpectrogram(n_fft=nfft, win_length=win_length,
                                                        window_fn=torch.hamming_window,
                                                        power=2,
                                                        normalized=False,
                                                        sample_rate=resampling)(record)

        else:
            spec = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=win_length,
                                                     window_fn=torch.hamming_window,
                                                     power=2,
                                                     normalized=False)(record)


        # spec = spec[0]
        spec = torch.log1p(spec)
        spec = torch.unsqueeze(spec, 0)
        # print(f"spec2: {spec.shape}")
        # spec = torch.unsqueeze(spec, dim=1)
        # db = F.AmplitudeToDB(top_db=60)
        # # print(record.shape)
        # spec = db(spec)
        # spec = torch.squeeze(spec, dim=1)
        path_index = str(path_index)
        return spec, record, label, path_index

    def __len__(self):

        """
        __len__ returns the len of the processed files

        :return: Number of processed files.
        """
        return len(self.files)
