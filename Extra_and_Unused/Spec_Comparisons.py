from torch.utils.data import DataLoader
from Jaguas_DataLoader_rainless import SoundscapeData
import matplotlib.pyplot as plt
from Modules.Utils import num_rows_cols
from Modules.Utils import plot_spectrogram
import torch
import torchaudio
import numpy as np

shape = [1, 2]
win = 300
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
                         audio_length=12, ext="wav", win_length=1028, spectrogram_type="sMel")
loader = DataLoader(dataset, batch_size=20)
if dataset.kwargs["spectrogram_type"] == "Mel":
    mel = True
else:
    mel = False

iterator = iter(loader)
a = next(iterator)
# a = next(iterator)
# a = next(iterator)
print(a[0].shape)
x, y = num_rows_cols(a[0].shape[0])
for i in range(a[0].shape[0]):
    plot_spectrogram(torch.log1p(a[0][i, 0, 0]), "torchaudio", numx_plots=x, numy_plots=y, i=i)
# plt.savefig(f"Spec_comparison_audio_sr_22050_{dataset.audio_length}_seconds_winlength_{dataset.win_length}_mel_{mel}_size_{a[0][0,0,0].shape}.pdf")
plt.show()
