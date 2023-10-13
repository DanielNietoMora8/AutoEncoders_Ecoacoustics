import numpy as np
import librosa
import librosa.display
import torch
import torchaudio
import os
from pathlib import Path

root_path = '/media/mirp_ai/Seagate Desktop Drive/Datos Rey Zamuro/Ultrasonido'
folders = os.listdir(root_path)
ext = "WAV"
files = []
for i in range(len(folders)):
    path_aux = "{}/{}".format(root_path, folders[i])
    files += list(Path(path_aux).rglob("*.{}".format(ext)))

print(len(files))

delimiter = "/"
resampling = 44100 // 2
path_index = str(files[2343])
recorder = str(path_index).split(delimiter)[-2]
record, sr = torchaudio.load(path_index)
record = torch.mean(record, dim=0, keepdim=True)
audio_len = 12 * resampling
record = torchaudio.transforms.Resample(sr, resampling)(record)
missing_padding = resampling * 60 - record.shape[1]
padding = torch.zeros([1, missing_padding])
record = torch.cat((record, padding), axis=1)
record = record[:, ::audio_len * (record.shape[1] // audio_len)]
record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
win_length = 255
nfft = int(np.round(1*win_length))
