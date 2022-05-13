# import torch
# import pandas as pd
# import soundfile as sf
# import torchaudio
# from pathlib import Path
# import os
# import sys
# from torch.utils.data import Dataset
# from scipy import signal
#
#
# root_path = "G:/Unidades compartidas/ConservacionBiologicaIA/Datos/Jaguas_2018"
# folders = os.listdir(root_path)
# print(len(folders))
# files = []
#
# for i in range(len(folders)):
#     path = Path(root_path+'/'+folders[i])
#     files += list(Path(path).rglob("*.{}".format("wav")))
# print("p1")
# print(files[0])
# record, sr = torchaudio.load(files[0])
# print(record.shape, sr)
# record = torch.mean(record, dim=0, keepdim=True)
# print(record.shape, sr)
# audio_len = 1*sr
# record = record[:, :audio_len * (record.shape[1] // audio_len)]
# print(record.shape)
# record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
# print(record.shape)


import matplotlib.pyplot as plt
import scipy.signal
import torch
from IPython.display import Audio, display
import torchaudio
import numpy as np
from Utils import plot_spectrogram
from torch.utils.data import DataLoader
from Jaguas_DataLoader import SoundscapeData
import sounddevice as sd
from Utils import play_audio
import cv2
import librosa


root_path = "G:/Unidades compartidas/ConservacionBiologicaIA/Datos/Porce_2019"

cuda = torch.device('cuda:0')
torch.cuda.empty_cache()
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

dataset = SoundscapeData(root_path=root_path, audio_length=59)
dataloader = DataLoader(dataset, batch_size=5)

dataiter = iter(dataloader)
S, record, sr = dataiter.next()
record_2 = torch.unsqueeze(record, dim=2)
win_length = 256
hop = int(np.round(win_length/1.5))*59
nfft = int(np.round(1*win_length))

print(f"hop:{hop}, nfft:{nfft}")
sxx = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=win_length,
                                        hop_length=hop,
                                        window_fn=torch.hamming_window,
                                        power=2,
                                        normalized=False,
                                        center=True)(record)

fig = plot_spectrogram(sxx[0, 0], "First_Try")
sd.play(record_2[0, 0, 0], 22050)

# plt.savefig('image.png', bbox_inches='tight', pad_inches=0)
#
# dim = (128, 128)
#
# im = cv2.imread("try2.png")
# resized = cv2.resize(im, dim, interpolation=cv2.INTER_LINEAR)
# cv2.imshow("re", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


record.shape
