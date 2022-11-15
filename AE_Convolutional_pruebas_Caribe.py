import os
import sys

import torch
import torch.nn as nn
from Models import ConvAE as Autoencoder_Convolutional
import torch.optim as optim
from torchvision import datasets, transforms
import torchaudio.transforms as ttf
import torchaudio.functional as F
from Dataloader import EcoData
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import librosa.display
import torch

"""
Test code using audio data and the EcoDataTesis dataloader.
"""

cuda = torch.device('cuda:0')
torch.cuda.empty_cache()

device = torch.device(cuda if torch.cuda.is_available() else "cpu")
# %% New dataloader

root_path = 'Shareddrives/ConservacionBiologicaIA/Datos'
labels_path = 'Shareddrives/ConservacionBiologicaIA/Datos/Acceso_Datos_Humboldt/ensayo.xlsx'
names_path = 'Shareddrives/ConservacionBiologicaIA/Datos/Acceso_Datos_Humboldt/RecordingsGuajiraTesisWeights.xlsx'

dataset = EcoData(root_path, labels_path, names_path, "wav")
dataloader = DataLoader(dataset, batch_size=1)
# dataiter = iter(dataloader)
# S, record, sr, features = dataiter.next()
#S_cuda = S.to(device)
# print(S.shape)


# %%
# plt.subplot(1,1,1)
# librosa.display.specshow(librosa.power_to_db(reconstructed.detach().numpy()[0,0], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
# librosa.display.specshow(librosa.power_to_db(S.detach().numpy()[0,0], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
# plt.show()

# %%
model = Autoencoder_Convolutional()
model_cuda = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
num_epochs = 3
outputs = []

# %%

for epoch in range(num_epochs):
    for (img, _, _, _) in dataloader:
        img_cuda = img.to(device)
        print(f"img shape:{img.shape}")
        recon = model_cuda(img_cuda)
        loss = criterion(recon, img_cuda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch+1}, Loss: {loss.item():.4f}')
    outputs.append((epoch, img, recon))

# %%

for k in range (0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9:
            break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >=9:
            break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item[0])
plt.show()


