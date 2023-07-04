import matplotlib.pyplot as plt
import os
import torch
import torchaudio
import numpy as np
from Modules.Utils import plot_spectrogram
from pathlib import Path
import librosa.display

cuda = torch.device('cuda:0')
torch.cuda.empty_cache()
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

root_path = "G:/Unidades compartidas/ConservacionBiologicaIA/Datos/Jaguas_2018"
folders = os.listdir(root_path)
print(len(folders))
files = []

for i in range(len(folders)):
    path = Path(root_path+'/'+folders[i])
    files += list(Path(path).rglob("*.{}".format("WAV")))

print("p1")
print(files[0])
record, sr = torchaudio.load(files[29])
audio_len = 12 * 22050
record = torch.mean(record, dim=0, keepdim=True)
record = torchaudio.transforms.Resample(sr, 22050)(record)
record = record[:, :1300950]
record = record[:, :audio_len * (record.shape[1] // audio_len)]
record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
record_numpy = record.numpy()

fig, ax = plt.subplots(figsize=(8, 6))
D = librosa.amplitude_to_db(np.abs(librosa.stft(record_numpy[0], n_fft=1028, win_length=516)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=22050, ax=ax)
ax.set(title='Spectral Domain - Fourier Transform')
ax.title.set_size(18)
ax.set_xlabel('time (seconds)', fontsize=14)
ax.set_ylabel('Frequency (Hz)', fontsize=14)
# ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
ax.label_outer()
print(D.shape)
plt.savefig("Fourier.pdf", format="pdf")
plt.show()

#%%
nfft = 1028
win_length = 1028
fig, ax = plt.subplots(figsize=(8, 6))
spec = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=win_length,
                                                 window_fn=torch.hamming_window,
                                                 power=2,
                                                 normalized=False)(record)

spec = torch.log1p(spec)
# spec = torch.unsqueeze(spec, 0)

img = librosa.display.specshow(spec.numpy()[0], y_axis='linear', x_axis='time',
                               sr=22050, ax=ax)

ax.set(title='Spectral Domain - Fourier Transform')
ax.title.set_size(18)
ax.set_xlabel('time (seconds)', fontsize=14)
ax.set_ylabel('Frequency (Hz)', fontsize=14)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
ax.label_outer()
plt.savefig("Fourier.pdf", format="pdf")
plt.show()
print("bien")

# fig, ax = plt.subplots(figsize=(20, 6))
# ax.plot(record[0], color="k")
# ax.set(title='Time Domain - Waveform')
# ax.title.set_size(18)
# ax.set_xlabel('Time', fontsize=18)
# ax.set_ylabel('Amplutide', fontsize=18)
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)
# ax.xaxis.label.set_size(18)
# ax.yaxis.label.set_size(18)
# plt.savefig("Waveform.pdf", format="pdf")
# plt.show()
#
# # plt.figure()
# # plt.plot(record[0])
# # plt.show()
