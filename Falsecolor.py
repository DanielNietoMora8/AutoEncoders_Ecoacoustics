import torch
import matplotlib.pyplot as plt
import torchaudio
from pathlib import Path
import os
from maad import sound, features
from maad.util import power2dB, plot2d
from skimage import transform
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
import numpy as np


root_path = "G:/Unidades compartidas/ConservacionBiologicaIA/Datos/Jaguas_2018"
folders = os.listdir(root_path)
print(len(folders))
files = []

for i in range(len(folders)):
    path = Path(root_path+'/'+folders[i])
    files += list(Path(path).rglob("*.{}".format("wav")))

for item in files[0:1]:
    s, fs = sound.load(item)
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=2048, noverlap=2)

    Sxx_db = power2dB(Sxx, db_range=70)
    Sxx_db = transform.rescale(Sxx_db, 0.5, anti_aliasing=True, multichannel=False)  # rescale for faster computation
    # plot2d(Sxx_db, **{'figsize': (4, 10), 'extent': ext})
    # plt.title('Spectrogram ' + str(item))
    # #plt.savefig("Spectrogram " + str(audio) + ".jpg")
    shape_im, params = features.shape_features_raw(Sxx_db, resolution='low')

    # Format the output as an array for decomposition
    print(Sxx_db.size)
    X = np.array(shape_im).reshape([len(shape_im), Sxx_db.size]).transpose()

    # Decompose signal using non-negative matrix factorization
    Y = NMF(n_components=3, init='random', random_state=0).fit_transform(X)

    # Normalize the data and combine the three NMF basis spectrograms and the
    # intensity spectrogram into a single array to fit the RGBA color model. RGBA
    # stands for Red, Green, Blue and Alpha, where alpha indicates how opaque each
    # pixel is.

    Y = MinMaxScaler(feature_range=(0, 1)).fit_transform(Y)
    intensity = 1 - (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min())
    plt_data = Y.reshape([Sxx_db.shape[0], Sxx_db.shape[1], 3])
    plt_data = np.dstack((plt_data, intensity))


#%%
    # fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    # for idx, ax in enumerate(axes):
    #     ax.imshow(plt_data[:, :, idx], origin='lower', aspect='auto',
    #               interpolation='bilinear')
    #     ax.set_axis_off()
    #     ax.set_title('Basis ' + str(idx + 1) + str(item))
    # plt.show()

#%%
    plt.figure(figsize=(10, 6))
    plt.imshow(plt_data, origin='lower', aspect='auto', interpolation='bilinear')
    # plt.set_axis_off()
    plt.title('False-color spectrogram ' + str(item))
    #plt.savefig("False-color  " + str(item) + ".jpg")
    plt.show()