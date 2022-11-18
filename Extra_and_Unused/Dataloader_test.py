import sys
import os
import torch
import torchaudio
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import numpy as np
from pathlib import Path
import librosa.display
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from Jaguas_DataLoader import SoundscapeData
from Models import ConvAE as AE
from AE_training_functions import TestModel, TrainModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "G:\\Mi unidad\\PhD_Thesis_Experiments\\DeepLearning\\AutoEncoders\\Project"
root_path = "ConservacionBiologicaIA/Datos/Jaguas_2018"
dataset = SoundscapeData(root_path, audio_length=12, ext="wav", win_length=1028)
dataset_train, dataset_test = random_split(dataset,
                                           [round(len(dataset)*0.3), len(dataset) - round(len(dataset)*0.3)],
                                           generator=torch.Generator().manual_seed(1024))


training_loader = DataLoader(dataset_train, batch_size=1)
test_loader = DataLoader(dataset_test, batch_size=1)

model_name = "AE_batch_size_56_num_hiddens_64__day_17_hour_0_final.pth"
config = torch.load(f'{root}\\config_AE_batch_size_56_num_hiddens_64__day_17_hour_0.pth')
model = AE(num_hiddens=config["num_hiddens"]).to(device)
dataset_test = torch.load(f'{root}\\temporal\\dataset_test_ae_jaguas_new')
dataset_train = torch.load(f'{root}\\temporal\\dataset_train_ae_jaguas_new')
model.load_state_dict(torch.load(f'{root}\\{model_name}', map_location=torch.device('cpu')))

iterator = iter(training_loader)
testing = TestModel(model, iterator, device=torch.device("cuda"))
originals, reconstructions, encodings, label, loss = testing.reconstruct()
encodings_size = encodings[0].shape

training_samples_list = []
delete_samples = []
training_samples_list_torch = torch.ones(6020, 5184).to("cuda")

for id, item in enumerate(dataset_train):
    print(f"id: {id + 1} of {len(dataset_train)}")
    model.to("cuda")
    try:
        originals, reconstructions, encodings, label, loss = testing.reconstruct()
    except:
        print(f"error id: {id}")
        delete_samples.append(id)
        continue

    encodings_size = encodings[0].shape
    encodings = encodings.to("cuda").detach()
    encodings = encodings.reshape(encodings.shape[0],
                                encodings.shape[1]*encodings.shape[2]*encodings.shape[3])
    encoding = encodings.squeeze(dim=0)
    # training_samples_list.append(encodings)
    training_samples_list_torch[id] = encodings
    if id % 500 == 0:
        torch.save(training_samples_list_torch, f"training_samples_list_torch_{id}.pth")


torch.save(training_samples_list_torch, "training_samples_list_torch.pth")







