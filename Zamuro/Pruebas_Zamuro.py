import numpy as np
import librosa
import librosa.display
import torch
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


path_index = files[2343]
recorder = str(path_index).split(delimiter)[-2]
recorder = int(recorder[1:3])
hour = int(str(path_index).split(delimiter)[-1].split("_")[2].split(".")[0][0:2])
minute = int(str(path_index).split(delimiter)[-1].split("_")[2].split(".")[0][2:4])
second = int(str(path_index).split(delimiter)[-1].split("_")[2].split(".")[0][4:6])
label = {"recorder": recorder,
         "hour": hour,
         "minute": minute,
         "second": second}
