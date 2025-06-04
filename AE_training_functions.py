import librosa
from scipy.io.wavfile import write
import os
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
    """
    Clase para evaluar modelos de autoencoder convolucional.

    Args:
        model (torch.nn.Module): Modelo entrenado.
        iterator (iter): Iterador de dataloader.
        num_views (int): Número de muestras para visualización.
        device (str): Dispositivo ('cuda' o 'cpu').
    """

    def __init__(self, model, iterator, num_views=8, device="cuda"):
        self.model = model
        self.iterator = iterator
        self.num_views = num_views
        self.device = device
        self.loss_history = {}

    def save_waveform(self, waveform, path):
        """Guarda un waveform como archivo WAV."""
        scaled = np.int16(waveform[0, 0] / np.max(np.abs(waveform[0, 0])) * 32767)
        write(f"{path}.wav", 22050, scaled)

    def plot_waveform(self, waveform, n_rows=4):
        """Grafica waveforms individuales."""
        fig, axs = plt.subplots(n_rows, figsize=(10, 6), constrained_layout=True)
        for i in range(n_rows):
            axs[i].plot(waveform[i, 0])
        plt.show()

    def waveform_generator(self, spec, n_fft=1028, win_length=1028):
        """
        Convierte un espectrograma a waveform.

        Returns:
            np.ndarray: Waveform reconstruido.
        """
        spec = spec.cdouble().to("cpu")
        transformation = audio_transform.InverseSpectrogram(n_fft=n_fft, win_length=win_length)
        waveform = transformation(spec)
        return waveform.cpu().detach().numpy()

    def plot_psd(self, waveform, n_wavs=1):
        """Grafica la densidad espectral de potencia (PSD)."""
        for i in range(n_wavs):
            plt.psd(waveform[i][0])
            plt.xlabel("Frecuencia")
            plt.ylabel("Densidad espectral")
            plt.show()

    def plot_reconstructions(self, originals, reconstructions):
        """
        Muestra imágenes originales y reconstruidas.

        Returns:
            matplotlib.figure.Figure: Figura generada.
        """
        combined = torch.cat((originals[:self.num_views], reconstructions[:self.num_views]), 0)
        grid = make_grid(combined, nrow=self.num_views, pad_value=20)
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.imshow(librosa.power_to_db(grid[1].cpu()), origin="lower")
        ax.axis("off")
        plt.show()
        return fig

    def reconstruct(self):
        """
        Reconstruye un lote de datos con el modelo.

        Returns:
            tuple: (originales, reconstrucciones, codificaciones, etiquetas, error, ruta)
        """
        self.model.eval()
        with torch.no_grad():
            originals, _, labels, path = next(self.iterator)
            originals = originals.view(-1, originals.shape[3], originals.shape[4]).unsqueeze(1).to(self.device)

            encodings = self.model.encoder(originals)
            reconstructions = self.model.decoder(encodings)

            # Para visualización
            loss = F.mse_loss(reconstructions, originals)

            return originals, reconstructions, encodings, labels, loss, path

    def run(self, plot=True, return_wave=True, show_wave=True, save_dir=None):
        """
        Ejecuta reconstrucción completa (opcional visualización y audio).

        Returns:
            tuple: (originales, reconstrucciones, codificaciones, etiquetas, error)
        """
        wave_orig = []
        wave_recon = []
        originals, reconstructions, encodings, labels, error, _ = self.reconstruct()

        if plot:
            self.plot_reconstructions(originals, reconstructions)

        if return_wave:
            wave_orig = self.waveform_generator(originals)
            wave_recon = self.waveform_generator(reconstructions)

            if show_wave:
                self.plot_waveform(wave_orig)
                self.plot_waveform(wave_recon)

            if save_dir:
                self.save_waveform(wave_orig, save_dir + "original_")
                self.save_waveform(wave_recon, save_dir + "reconstruction_")

        return originals, reconstructions, encodings, labels, error


class TrainModel:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def wandb_init(self, config, keys=["batch_size", "num_hiddens"]):
        """Inicializa wandb para monitoreo del entrenamiento."""
        run_name = config.get("architecture", "model") + "_"
        run_name += "_".join(f"{key}_{config.get(key, 'NA')}" for key in keys)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name += f"_run_{timestamp}"

        try:
            wandb.login()
            wandb.finish()  # Termina sesiones previas si existen
            wandb.init(project=config["project"], config=config)
            wandb.run.name = run_name
            wandb.run.save()
            wandb.watch(self.model, F.mse_loss, log="all", log_freq=1)
            return True, run_name
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            return False, run_name

    def wandb_log(self, metrics: dict):
        """Registra métricas en wandb."""
        wandb.log(metrics)

    def train_one_step(self, data, optimizer):
        """Realiza una iteración de entrenamiento."""
        data = data.view(-1, data.shape[3], data.shape[4]).unsqueeze(1).to(self.device)

        self.model.train()
        optimizer.zero_grad()
        output = self.model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        return loss.item()

    def log_reconstructions(self, model, data_iter, step, mode="test"):
        """
        Realiza reconstrucciones y las registra en wandb.

        Args:
            model (torch.nn.Module): Modelo a evaluar.
            data_iter (iterator): Iterador de dataloader.
            step (int): Paso de entrenamiento actual.
            mode (str): "train" o "test" para etiquetar los logs.
        """
        try:
            if mode == "train":
                model.train()
            else:
                model.eval()

            test_instance = TestModel(model, data_iter, num_views=8, device=self.device)
            originals, reconstructions, encodings, labels, loss, path = test_instance.reconstruct()
            fig = test_instance.plot_reconstructions(originals, reconstructions)
            image = wandb.Image(fig, caption=f"{mode.capitalize()} Recon Error: {loss.item():.4f}")
            self.wandb_log({
                f"{mode}/recon_examples": image,
                f"{mode}/recon_loss": loss.item(),
                "step": step
            })

            # # === Guardar métricas localmente ===
            # time_now = datetime.datetime.now()
            # folder_name = f'temporal_zamuro/models/New_paper/model_{wandb.run.name}'
            # os.makedirs(folder_name, exist_ok=True)
            #
            # metrics = {
            #     "mode": mode,
            #     "step": step,
            #     "loss": loss.item(),
            #     "timestamp": time_now.isoformat()
            # }
            #
            # save_path = os.path.join(folder_name, f"{mode}_metric_step_{step}.pt")
            # torch.save(metrics, save_path)
            # print(f"{mode.capitalize()} metrics saved to {save_path}")

        except Exception as e:
            print(f"Error logging {mode} reconstructions: {e}")

    def forward(self, training_loader, test_loader, config):
        wandb_enabled, run_name = self.wandb_init(config)
        optimizer = config["optimizer"]
        scheduler = config["scheduler"]
        logs = []

        self.loss_history = {}  # Inicializar estructura para guardar pérdidas por época

        for epoch in range(config["num_epochs"]):
            train_iter = iter(training_loader)
            test_iter = iter(test_loader)
            self.loss_history[epoch] = {"train": [], "test": []}  # Inicializar pérdidas para esta época

            for step in range(config["num_training_updates"]):
                try:
                    data, *_ = next(train_iter)
                    step_count = step + epoch * config["num_training_updates"]

                    # === Entrenamiento ===
                    loss = self.train_one_step(data, optimizer)
                    print(
                        f"Epoch {epoch + 1}/{config['num_epochs']} - Step {step + 1}/{config['num_training_updates']} - Loss: {loss:.4f}")
                    self.loss_history[epoch]["train"].append(loss)
                    self.wandb_log({"train/loss": loss, "step": step_count})

                    # === Evaluación con test ===
                    if wandb_enabled:
                        test_iter_temp = iter(test_loader)
                        test_instance = TestModel(self.model, test_iter_temp, num_views=8, device=self.device)
                        _, _, _, _, test_loss, _ = test_instance.reconstruct()
                        test_loss_value = test_loss.item()
                        self.loss_history[epoch]["test"].append(test_loss_value)
                        self.wandb_log({"test/loss": test_loss_value, "step": step_count})

                        # === Cada 50 pasos: log de reconstrucciones ===
                        if (step + 1) % 50 == 0:
                            recon_step = step // 50
                            self.log_reconstructions(self.model, test_iter, recon_step, mode="test")
                            train_iter_temp = iter(training_loader)
                            self.log_reconstructions(self.model, train_iter_temp, recon_step, mode="train")

                except Exception as e:
                    print(f"Training step failed: {e}")
                    logs.append(e)

            scheduler.step()
            torch.cuda.empty_cache()
            self._save_model(epoch, run_name)

            # === Guardar historia de pérdidas por época ===
            loss_save_dir = os.path.join("temporal_zamuro", "models", "New_paper")
            os.makedirs(loss_save_dir, exist_ok=True)
            loss_save_path = os.path.join(loss_save_dir, f"loss_history_{run_name}.pt")
            torch.save(self.loss_history, loss_save_path)
            print(f"Saved cumulative loss history to {loss_save_path}")

            clear_output(wait=True)
            print(f"Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

        wandb.finish()
        return self.model, logs, run_name


    def _save_model(self, epoch, run_name):
        """Guarda el modelo actual en una carpeta específica."""
        time = datetime.datetime.now()
        folder_name = f'temporal_zamuro/models/New_paper/model_{run_name}'
        save_dir = os.path.join('temporal_zamuro', 'models', folder_name)
        os.makedirs(save_dir, exist_ok=True)

        filename = f'epoch_{epoch + 1}_training.pth'
        save_path = os.path.join(save_dir, filename)

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))