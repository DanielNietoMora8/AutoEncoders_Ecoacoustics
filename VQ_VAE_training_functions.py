import librosa
from scipy.io.wavfile import write
from six.moves import xrange
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torchaudio.transforms as audio_transform
import torch.nn.functional as F
from torch import nn
import wandb
from IPython.display import clear_output
from wandb import AlertLevel
import datetime
from datetime import timedelta

class TestModel:

    def __init__(self, model, iterator, num_views=8, device="cuda"):
        self._model = model
        self._iterator = iterator
        self.num_views = num_views
        self.device = device

    def plot_waveform(self, waveform, n_rows=2, directory=None):
        fig, axs = plt.subplots(n_rows, figsize=(10, 6))
        for i in range(len(waveform)):
            axs[i].plot(waveform[i, 0])
            if directory != None:
                scaled = np.int16(waveform[i, 0] / np.max(np.abs(waveform[i, 0])) * 32767)
                write(directory + str(i) + '.wav', 22050, scaled)
        plt.show()

    def waveform_generator(self, spec, n_fft=1028, win_length=1028, base_win=256, plot=False):
        spec = spec.cdouble()
        spec = spec.to("cpu")
        hop_length = int(np.round(base_win / win_length * 172.3))
        transformation = audio_transform.InverseSpectrogram(n_fft=n_fft, win_length=win_length)
        waveform = transformation(spec)
        waveform = waveform.cpu().detach().numpy()
        return waveform

    def plot_psd(self, waveform):
        for wave in waveform:
            plt.psd(wave)

    def plot_reconstructions(self, imgs_original, imgs_reconstruction, num_views: int = 8):
        output = torch.cat((imgs_original[0:self.num_views], imgs_reconstruction[0:self.num_views]), 0)
        img_grid = make_grid(output, nrow=self.num_views, pad_value=20)
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.imshow(librosa.power_to_db(img_grid[1, :, :].cpu()), origin="lower")
        ax.axis("off")
        plt.show()
        return fig

    def reconstruct(self):
        self._model.eval()
        (valid_originals, _, label, path) = next(self._iterator)
        valid_originals = torch.reshape(valid_originals, (valid_originals.shape[0] * valid_originals.shape[1]
                                                          * valid_originals.shape[2], valid_originals.shape[3],
                                                          valid_originals.shape[4]))
        valid_originals = torch.unsqueeze(valid_originals, 1)

        valid_originals = valid_originals.to(self.device)

        vq_output_eval = self._model._pre_vq_conv(self._model._encoder(valid_originals))
        _, valid_quantize, _, _ = self._model._vq_vae(vq_output_eval)

        valid_encodings = self._model._encoder(valid_originals)
        # print(valid_quantize.shape)

        valid_reconstructions = self._model._decoder(valid_quantize)

        recon_error = F.mse_loss(valid_originals, valid_reconstructions)

        return valid_originals, valid_reconstructions, valid_encodings, label, recon_error, path

    def run(self, plot=True, wave_return=True, wave_plot=True, directory=None):
        wave_original = []
        wave_reconstructions = []
        originals, reconstructions, error = self.reconstruct()
        if plot:
            self.plot_reconstructions(originals, reconstructions)
        if wave_return:
            wave_original = self.waveform_generator(originals)
            wave_reconstructions = self.waveform_generator(reconstructions)
            if wave_plot:
                self.plot_waveform(wave_original, len(wave_original), directory="originals")
                self.plot_waveform(wave_reconstructions, len(wave_reconstructions), directory="reconstructions")

        return originals, reconstructions, wave_original, wave_reconstructions, error


class TrainModel:

    def __init__(self, model):
        self._model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)
        print(self.device)

    def wandb_init(self, config, keys=["batch_size", "num_embeddings", "embedding_dim"]):
        try:
            run_name = str(config["architecture"]+"_")
            for key in keys:
                if key in config.keys():
                    run_name = run_name + key + "_" + str(config[key]) + "_"
                else:
                    run_name = run_name + str(key)

            wandb.login()
            wandb.finish()
            wandb.init(project=config["project"], config=config)
            wandb.run.name = run_name
            wandb.run.save()
            wandb.watch(self._model, F.mse_loss, log="all", log_freq=1)
            is_wandb_enable = True
        except Exception as e:
            print(e)
            is_wandb_enable = False

        return is_wandb_enable, run_name

    def wandb_logging(self, dict):
        for keys in dict:
            wandb.log({keys: dict[keys]})

    def forward(self, training_loader, test_loader, config):
        #         iterator = iter(test_loader)
        wandb_enable, run_name = self.wandb_init(config)
        optimizer = config["optimizer"]
        scheduler = config["scheduler"]

        train_res_recon_error = []
        train_res_perplexity = []
        logs = []
        best_loss = 10000

        for epoch in range(config["num_epochs"]):
            iterator = iter(test_loader)
            iterator_train = iter(training_loader)
            for i in xrange(config["num_training_updates"]):
                self._model.train()
                try:
                    (data, _, _, _) = next(iterator_train)
                except Exception as e:
                    print("error")
                    print(e)
                    logs.append(e)
                    continue

                data = torch.reshape(data,
                                     (data.shape[0] * data.shape[1] * data.shape[2], data.shape[3], data.shape[4]))
                data = torch.unsqueeze(data, 1)
                data = data.to(self.device)
                # print(data.shape)

                optimizer.zero_grad()
                vq_loss, data_recon, perplexity = self._model(data)
                # print(data_recon.shape)

                recon_error = F.mse_loss(data_recon, data)  # / data_variance
                loss = recon_error + vq_loss
                loss.backward()

                optimizer.step()
                print(
                    f'epoch: {epoch + 1} of {config["num_epochs"]} \t iteration: {(i + 1)} of {config["num_training_updates"]} \t loss: {np.round(loss.item(), 4)} \t recon_error: {np.round(recon_error.item(), 4)} \t vq_loss: {np.round(vq_loss.item(), 4)}')
                dict = {"loss": loss.item(),
                        "perplexity": perplexity.item(),
                        "recon_error": recon_error,
                        "vq_loss": vq_loss}
                #                 step = epoch*config["num_training_updates"] + i
                self.wandb_logging(dict)

                period = 50
                if (i + 1) % period == 0:
                    try:
                        test_ = TestModel(self._model, iterator, 8)
                        # torch.save(model.state_dict(),f'model_{epoch}_{i}.pkl')
                        originals, reconstructions, encodings, label, test_error, path = test_.reconstruct()
                        fig = test_.plot_reconstructions(originals, reconstructions, 8)
                        images = wandb.Image(fig, caption=f"recon_error: {np.round(test_error.item(), 4)}")
                        self.wandb_logging({"examples": images, "step": (i + 1) // period})

                    except Exception as e:
                        print("error")
                        logs.append(e)
                        continue
                else:
                    pass

            scheduler.step()
            torch.cuda.empty_cache()
            time = datetime.datetime.now()
            torch.save(self._model.state_dict(),
                       f'temporal_zamuro/models/model_{run_name}_month_{time.month}_day_{time.day}_hour_{time.hour}_epoch_{epoch + 1}_training.pth')
            clear_output()
            print(optimizer.state_dict()["param_groups"][0]["lr"])

        wandb.finish()
        return self._model, logs, run_name