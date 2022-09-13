import torch
import pandas as pd
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset
from scipy import signal
import torchaudio

class EcoDataTesis(Dataset):

    """
    EcoDataTesisReduced is the main class used by the dataloader function allowing access to acoustic data,
    and then training and testing a deep learning model. This version allows to define the audio length
    to return.
    """

    def __init__(self, root_path: str, path_labels: str, path_names: str, audio_length: int, ext="wav"):

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
        self.files = []
        self.pd_labels = pd.read_excel(path_labels)
        self.pd_files = pd.read_excel(path_names)

        for i in range(len(self.pd_files['Carpetas'].values)):
            path_aux = "{}/{}".format(root_path, self.pd_files['Carpetas'].iloc[i])
            self.files += list(Path(path_aux).rglob("*.{}".format(ext)))

    def __getitem__(self, index):

        """
        Function used to return audios and spectrograms based on the batch size. Here it is searched and processed the
        files to return each audio with it respective.

        :param index: index indicates the number of data to return.
        :returns:
            :spectrogram: Spectrogram of the indexed audios.
            :type spectrogram: torch.tensor
            :record: array representation of the indexed audios.
            :type record: numpy.array
            :sr: Sample rate.
            :type sr: int
            :labels: Audio labels from the info file.
            :type labels: Dataframe.

        """

        path_index = self.files[index]
        split_filename = str(path_index).split("/")
        split_filename = split_filename[len(self.root_path.split("/")):]
        if split_filename[0] == 'Guajira_2016':
            serial = int(split_filename[2].split("_")[-1][2:])
            date = int(split_filename[-1].split('_')[1])
            labels_file = self.pd_labels[self.pd_labels['Departamento'] == split_filename[0]]

        elif split_filename[0] == 'Caribe':
            serial = int(split_filename[2])
            date = int(split_filename[3])
            labels_file = self.pd_labels.copy()

        labels_file = labels_file[labels_file['Serial Songmeter'] == serial]

        for i in range(len(labels_file)):
            date_ini = int(labels_file["Fecha Inicio"].iloc[i])
            date_end = int(labels_file["Fecha Final"].iloc[i])
            if date_ini <= date <= date_end:
                features = {"transform": labels_file["Transformación"].iloc[i],
                            "Latitud": labels_file["Latitud"].iloc[i],
                            "Longitud": labels_file["Longitud"].iloc[i],
                            "Permanencia": labels_file["Permanencia"].iloc[i]}
                break
            else:
                continue

        record, sr = torchaudio.load(path_index)
        audio_len = self.audio_length * sr
        record = record[:, :audio_len * (record.shape[1] // audio_len)]

        record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
        spec = torchaudio.transforms.Spectrogram(n_fft=2048)(record)
        return spec, record, sr, features

    def __len__(self):

        """
        __len__ returns the len of the processed files

        :return: Number of processed files.
        """
        return len(self.files)