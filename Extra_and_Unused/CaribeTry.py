import torch
import pandas as pd
import soundfile as sf
import torchaudio
from pathlib import Path
import os
import sys
from torch.utils.data import Dataset
from scipy import signal

cuda = torch.device('cuda:0')
torch.cuda.empty_cache()


root_path = "G:/Unidades compartidas/ConservacionBiologicaIA/Datos/Caribe/2016/5071/20161221"
files = os.listdir(root_path)
print(files[0])

path_aux = '{}/{}'.format(root_path, files[0])
print(path_aux)
record, sr = torchaudio.load(path_aux)
audio_len = 1*sr
print(record.shape)
record = record[:, :audio_len * (record.shape[1] // audio_len)]
print(f'first length: {record.shape}')
record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
print(f'second length: {record.shape}')

