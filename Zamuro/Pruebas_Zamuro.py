import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from Modules.Utils import plot_spectrogram
import torch
import torchaudio
import os
from pathlib import Path
import pandas as pd

dataframe_path = ('/home/mirp_ai/Documents/Daniel_Nieto/PhD/AutoEncoders_Ecoacoustics/'
                  'Zamuro/Complementary_Files/zamuro_audios.csv')
df_audios = pd.read_csv(dataframe_path)
print(len(df_audios))

filters = {"rain_FI": "NO"}
for key in filters.keys():
    df_audios = df_audios[df_audios[key] == filters[key]]
print(len(df_audios))
folders = list(df_audios["field_number_PR"])
files = list(df_audios["Filename"])


#%%
# dataframe_path = ('/media/mirp_ai/Seagate Desktop Drive/Datos Rey Zamuro/Ultrasonido')
# folders = os.listdir(dataframe_path)
# ext = "WAV"
# files = []
# for i in range(len(folders)):
#     path_aux = "{}/{}".format(root_path, folders[i])
#     files += list(Path(path_aux).rglob("*.{}".format(ext)))
#
# print(len(files))

#%%
# delimiter = "/"
# resampling = 44100 // 2
# path_index = files[2384]
# recorder = folders[2384]
# record2, sr = torchaudio.load('/media/mirp_ai/Seagate Desktop Drive/Datos Rey Zamuro/Ultrasonido/'+recorder+"/"+path_index)
#
# #%%
# record = torch.mean(record2, dim=0, keepdim=True)
# audio_len = 12 * resampling
# record = torchaudio.transforms.Resample(sr, resampling)(record)
# print("record 1", record.shape)
# missing_padding = resampling * 60 - record.shape[1]
# padding = torch.zeros([1, missing_padding])
# record = torch.cat((record, padding), axis=1)
# print("record 2", record.shape)
# record = record[:, :audio_len * (record.shape[1] // audio_len)]
# print("record 3", record.shape)
# record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
# print("record 4", record.shape)
# win_length = 1028
# nfft = int(np.round(win_length))
# spec = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=win_length,
#                                          window_fn=torch.hamming_window,
#                                          power=2,
#                                          normalized=False)(record)
#
# fig = plot_spectrogram(spec[0], nfft=nfft, title="torchaudio")
# plt.show()

#%%

from Zamuro_DataLoader import SoundscapeData
from torch.utils.data import random_split
from torch.utils.data import DataLoader

dataset = SoundscapeData('/media/mirp_ai/Seagate Desktop Drive/Datos Rey Zamuro/Ultrasonido/',
                         dataframe_path="Complementary_Files/zamuro_audios.csv",
                         audio_length=12, ext="wav",
                         win_length=1028, filters=filters)
dataset_train, dataset_test = random_split(dataset,
                                           [round(len(dataset)*0.98), len(dataset) - round(len(dataset)*0.98)],
                                           generator=torch.Generator().manual_seed(1024))

loader = DataLoader(dataset_train, batch_size=10)
iterator = iter(loader)
a, b, c, d = next(iterator)
print(a.shape)
