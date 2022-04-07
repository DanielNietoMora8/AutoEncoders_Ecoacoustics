import librosa
import torchaudio
import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display
from scipy import signal


def Spectrogram(files: list, module: str = "librosa", sr: int = 48000, n_fft: int = 1024, window_length: int = 1024):

    if module == 'librosa':
        for i in files:
            print(f'processing image: {i}')
            record, _ = librosa.load(str(i))
            hop = int(np.round(window_length/4))
            n_fft = int(np.round(2*window_length))
            window = signal.windows.hamming(window_length, sym=False)
            S = np.abs(librosa.stft(record, n_fft=window_length, hop_length=hop, window=window,
                                    win_length=window_length))
            fig, ax1 = plt.subplots()
            img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis="hz", ax=ax1)
            ax1.set_title('Power spectrogram')
            fig.colorbar(img, ax=ax1, format="%+2.0f dB")
            plt.show()

    elif module == 'torch':
        for i in files:
            record, sr = torchaudio.load(str(i))
            spec = torchaudio.transforms.Spectrogram(n_fft=n_fft)(record)
            plt.figure()
            plt.imshow(spec.log2()[0, :].numpy())
            plt.show()

    else:
        print(f"Module {module} does not able to use ")
