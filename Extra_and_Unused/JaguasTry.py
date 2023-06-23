import matplotlib.pyplot as plt
import os
import torch
import torchaudio
import numpy as np
from Modules.Utils import plot_spectrogram
from Modules.Utils import num_rows_cols
from pathlib import Path
import librosa.display
from torch.utils.data import DataLoader

#%%
from Jaguas_DataLoader_rainless import SoundscapeData

shape = [1, 2]
win = 20
while (shape[0] != shape[1]) & (shape[1]>shape[0]):
    if shape[1]/shape[0] > 4:
        win += 32
    elif shape[1]/shape[0] > 2:
        win += 24
    elif shape[1]/shape[0] > 1.2:
        win += 4
    elif shape[1]/shape[0] > 1.1:
        win += 4
    else:
        win += 1
    dataset = SoundscapeData(root_path="ConservacionBiologicaIA/Datos/Jaguas_2018",
                             dataframe_path="Jaguas\Complementary_Files\Audios_Jaguas\G04.csv",
                             audio_length=1, ext="wav", win_length=win, spectrogram_type="sMel")
    loader = DataLoader(dataset, batch_size=1)
    iterator = iter(loader)
    a = next(iterator)
    shape = a[0][0, 0, 0].shape
    print(shape[0], shape[1], win)
if shape[0] == shape[1]:
    print(f"win length found: {win}, shape:{shape}")

#%%
from Jaguas_DataLoader_rainless import SoundscapeData
dataset = SoundscapeData(root_path="ConservacionBiologicaIA/Datos/Jaguas_2018",
                         dataframe_path="Jaguas\Complementary_Files\Audios_Jaguas\G04.csv",
                         audio_length=6, ext="wav", win_length=514, spectrogram_type="sMel")
loader = DataLoader(dataset, batch_size=20)
if dataset.kwargs["spectrogram_type"] == "Mel":
    mel = True
else:
    mel = False

iterator = iter(loader)
a = next(iterator)
a = next(iterator)
a = next(iterator)
print(a[0].shape)
x, y = num_rows_cols(a[0].shape[0])
for i in range(a[0].shape[0]):
    plot_spectrogram(a[0][i, 0, 0], "torchaudio", numx_plots=x, numy_plots=y, i=i)
# plt.savefig(f"Spec_comparison_audio_sr_22050_{dataset.audio_length}_seconds_winlength_{dataset.win_length}_mel_{mel}_size_{a[0][0,0,0].shape}.pdf")
plt.show()
#%%
import pandas as pd
df_folders = pd.read_csv("Jaguas\Complementary_Files\Audios_Jaguas\G04.csv")
files = df_folders[df_folders["Intensity_Category"] == "No_rain"]
len(files)


#%%
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
resampling = 22050
record = torchaudio.transforms.Resample(sr, resampling)(record)
missing_padding = resampling * 60 - record.shape[1]
padding = torch.zeros([1, missing_padding])
record = torch.cat((record, padding), axis=1)
record = record[:, :audio_len * (record.shape[1] // audio_len)]
record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))

fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(record[0], color="k")
ax.set(title='Time Domain - Waveform')
ax.title.set_size(18)
ax.set_xlabel('Time', fontsize=18)
ax.set_ylabel('Amplutide', fontsize=18)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
# plt.savefig("Waveform.pdf", format="pdf")
plt.show()

# plt.figure()
# plt.plot(record[0])
# plt.show()




#%%
base_win = 256
win_length = 2047
hop = int(np.round(base_win/win_length * 172.3 * 59))
# hop = int(np.round(win_length / 95.4))*59
nfft = int(np.round(1*win_length))


sxx = torchaudio.transforms.Spectrogram(n_fft=1028,
                                        win_length=1028,
                                        window_fn=torch.hamming_window,
                                        power=2,
                                        normalized=False,
                                        center=True)(record)

fig = plot_spectrogram(sxx[0], "torchaudio")

record_2 = record.detach().numpy()
print(record_2.shape)
D = librosa.stft(record_2[0], n_fft=512, hop_length=64)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure()
plt.title("librosa")
librosa.display.specshow(S_db, sr=sr)
plt.show()
#%%
plt.figure()
plt.imshow(S_db)
plt.show()
#%%
import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.ex('choice'), duration=25)
fig, ax = plt.subplots(figsize=(8, 6))
D = librosa.amplitude_to_db(np.abs(librosa.stft(record_2[0], n_fft=512, hop_length=8)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=sr, ax=ax)
ax.set(title='Spectral Domain - Fourier Transform')
ax.title.set_size(18)
ax.set_xlabel('Hour', fontsize=14)
ax.set_ylabel('Frequency (Hz)', fontsize=14)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
ax.label_outer()
print(D.shape)
plt.savefig("Fourier.pdf", format="pdf")
plt.show()

#%%

chroma = librosa.feature.chroma_stft(S=D, sr=sr)
fig, ax = plt.subplots(figsize=(8, 6))
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
ax.set(title='Chromagram')
ax.title.set_size(18)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.savefig("Chroma.pdf", format="pdf")
plt.show()

#%%
m_slaney = librosa.feature.mfcc(y=record_2[0], sr=sr, dct_type=2)
fig, ax = plt.subplots(figsize=(8,6))
img1 = librosa.display.specshow(m_slaney, x_axis='time', ax=ax)
ax.set(title='Spectral Domain - MFCC')
ax.title.set_size(18)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.savefig("MFCC.pdf", format="pdf")
plt.show()

#%%
hop_length = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                      hop_length=hop_length)
# Compute global onset autocorrelation
ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
ac_global = librosa.util.normalize(ac_global)
# Estimate the global tempo for display purposes
tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                           hop_length=hop_length)[0]
fig, ax = plt.subplots(figsize=(8, 6))
librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='tempo', cmap='magma',
                         ax=ax)
ax.set(title='Time Domain - Tempogram')
x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
                num=tempogram.shape[0])

ax.legend(loc='upper right')
ax.title.set_size(18)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.savefig("Tempogram.pdf", format="pdf")
plt.show()

#%%
record, sr = librosa.load(files[104], duration=12)
resampling = 22050
record = librosa.resample(record, orig_sr=sr, target_sr=resampling)
win_length = 1028
nfft = int(np.round(1*win_length))
spec = (np.abs(librosa.stft(record, n_fft=1028, hop_length=nfft//2)))
print(spec)
h, p = librosa.decompose.hpss(spec)
print(type(h))
h = np.expand_dims(h, axis=0)
p = np.expand_dims(p, axis=0)
spec = np.expand_dims(spec, axis=0)
h = torch.from_numpy(h)
p = torch.from_numpy(p)
plt.figure()
plt.imshow(spec[0], origin="lower", vmin=0, vmax=2)
plt.show()
plt.figure()
plt.imshow(h[0], origin="lower", vmin=0, vmax=2)
plt.show()
plt.figure()
plt.imshow(p[0], origin="lower", vmin=0, vmax=2)
plt.show()

# y_harm = librosa.istft(h)
# y_perc = librosa.istft(p)
# spec_H = torch.from_numpy(h)
# spec_P = torch.from_numpy(p)

