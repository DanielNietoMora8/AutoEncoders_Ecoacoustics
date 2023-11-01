import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from Modules.Utils import plot_spectrogram
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
record2, sr = torchaudio.load(path_index)
record = torch.mean(record2, dim=0, keepdim=True)
audio_len = 12 * resampling
record = torchaudio.transforms.Resample(sr, resampling)(record)
print("record 1", record.shape)
missing_padding = resampling * 60 - record.shape[1]
padding = torch.zeros([1, missing_padding])
record = torch.cat((record, padding), axis=1)
print("record 2", record.shape)
record = record[:, :audio_len * (record.shape[1] // audio_len)]
print("record 3", record.shape)
record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
print("record 4", record.shape)
win_length = 1028
nfft = int(np.round(win_length))
spec = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=win_length,
                                         window_fn=torch.hamming_window,
                                         power=2,
                                         normalized=False)(record)

fig = plot_spectrogram(spec[0], nfft=nfft, title="torchaudio")
plt.show()
