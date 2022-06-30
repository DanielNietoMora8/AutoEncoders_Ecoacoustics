import os
import torch
from pathlib import Path
import torchaudio

root_path = "G:/Unidades compartidas/ConservacionBiologicaIA/Datos/Jaguas_2018"
folders = os.listdir(root_path)
print(len(folders))
files = []

for i in range(len(folders)):
    path = Path(root_path+'/'+folders[i])
    files += list(Path(path).rglob("*.{}".format("wav")))
print("p1")
print(files[0])
record, sr = torchaudio.load(files[0])
print(record.shape, sr)
record = torch.mean(record, dim=0, keepdim=True)
print(record.shape, sr)
audio_len = 1*sr
record = record[:, :audio_len * (record.shape[1] // audio_len)]
print(record.shape)
record = torch.reshape(record, (record.shape[1] // audio_len, audio_len))
print(record.shape)