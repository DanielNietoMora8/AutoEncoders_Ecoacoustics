import matplotlib.pyplot as plt
import os
import torch
import torchaudio
import numpy as np
from Modules.Utils import plot_spectrogram
from torch.utils.data import DataLoader
from Jaguas_DataLoader import SoundscapeData
from pathlib import Path
import sounddevice as sd
import librosa
import librosa.display
import cv2

cuda = torch.device('cuda:0')
torch.cuda.empty_cache()
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

root_path = "G:/Unidades compartidas/ConservacionBiologicaIA/Datos/Porce_2019"
folders = os.listdir(root_path)
print(len(folders))
files = []

for i in range(len(folders)):
    path = Path(root_path+'/'+folders[i])
    files += list(Path(path).rglob("*.{}".format("wav")))

print("p1")
print(files[0])
record, sr = torchaudio.load(files[0])
audio_len = 59 * 22050
record = torch.mean(record, dim=0, keepdim=True)
record = torchaudio.transforms.Resample(sr, 22050)(record)
record = record[:, :1300950]
record = record[:, :audio_len * (record.shape[1] // audio_len)]
record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))


#%%
base_win = 256
win_length = 2047
hop = int(np.round(base_win/win_length * 172.3 * 59))
# hop = int(np.round(win_length / 95.4))*59
nfft = int(np.round(1*win_length))


sxx = torchaudio.transforms.Spectrogram(n_fft=nfft,
                                        win_length=win_length,
                                        hop_length=hop,
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

plt.figure()
plt.imshow(S_db)
plt.show()

#%%
import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.ex('choice'), duration=25)
fig, ax = plt.subplots()
D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=8)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=sr, ax=ax)
ax.set(title='Linear-frequency power spectrogram')
ax.label_outer()
print(D.shape)
plt.show()

#%%

y, sr = librosa.load(files[0], duration=12)
D = librosa.stft(y, n_fft=1028, hop_length=514)
H, P = librosa.decompose.hpss(D)

y_harm = librosa.istft(H)
y_perc = librosa.istft(P)
spec_H = torch.from_numpy(H)
spec_P = torch.from_numpy(P)

