import torch
import librosa
import torchaudio
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class SoundscapeData(Dataset):

    """
    JaguasData is the main class used by the dataloader function allowing access to acoustic data of jaguas,
    and then training and testing a deep learning model. This version allows to define the audio length
    to return.
    """

    def __init__(self, root_path: str, audio_length: int, ext: str = "wav", win_length: int = 255):

        """
        This function is used to initialize the Dataloader, here path and root of files are defined.

        :param root_path: Main root of all files.
        :type root_path: str
        :param path_labels: Path of the unique file containing audios information.
        :type path_labels: str
        :param path_names: Path of a file that contains audios root.
        :type path_names: str
        :param ext: Audios extension (ex: .wav)
        """

        self.audio_length = audio_length
        self.root_path = root_path
        self.win_length = win_length
        self.folders = os.listdir(root_path)
        self.files = []
        for i in range(len(self.folders)):
            path_aux = "{}/{}".format(root_path, self.folders[i])
            self.files += list(Path(path_aux).rglob("*.{}".format(ext)))

    def __getitem__(self, index):

        """
        Function used to return audios and spectrograms based on the batch size. Here it is searched and processed the
        files to return each audio with it respective.

        :param index: index indicates the number of data to return.
        :returns:
            :spec: Spectrogram of the indexed audios.
            :type spec: torch.tensor
            :record: array representation of the indexed audios.
            :type record: numpy.array
            :sr: Sample rate.
            :type sr: int
            :features: Audio labels from the info file.
            :type features: Dataframe.

        """
        path_index = self.files[index]
        label = str(path_index).split("/")[-2]
        record, sr = librosa.load(path_index, duration=self.audio_length, mono=True)
        resampling = 22050
        record = librosa.resample(record, orig_sr=sr, target_sr=resampling)
        win_length = self.win_length
        nfft = int(np.round(1*win_length))
        spec = (np.abs(librosa.stft(record, n_fft=nfft, hop_length=nfft//2)))
        # spec = librosa.amplitude_to_db(np.abs(spec),
        #                                ref=np.max)
        h, p = librosa.decompose.hpss(spec)
        h = np.expand_dims(h, axis=0)
        p = np.expand_dims(p, axis=0)
        h = torch.from_numpy(h)
        p = torch.from_numpy(p)
        #spec = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=win_length,
        #                                          window_fn=torch.hamming_window,
        #                                          power=2,
        #                                          normalized=False)(record)
        # spec_h = torch.log1p(torch.from_numpy(h))
        # spec_p = torch.log1p(torch.from_numpy(p))

        return h, record, label

    def __len__(self):

        """
        __len__ returns the len of the processed files

        :return: Number of processed files.
        """
        return len(self.files)
