import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.cm as cm
import pandas as pd
import torch
import joypy
from webencodings import LABELS


def plot_silhouette(X, cluster_labels, n_clusters, silhouette_avg, method=None,
                    extra=None, save=False, root=None):
    fig, ax1 = plt.subplots(figsize=(12, 12))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    if not extra:
        extra= ""
    else:
        pass

    if save and root is not None:
        plt.savefig(f"{root}/Silhouette_plot_{n_clusters}_{extra}.pdf", format="pdf")
        plt.show()
    else:
        print("Ploted!")
        pass


def plot_centroids(cluster_centers, model, extra=None, save=True, root=None):
    plt.figure(figsize=(18, 18))
    model._model.to("cpu")
    for i, spec in enumerate(cluster_centers):
        encodings = spec.reshape(64, 9, 9)
        encodings = torch.tensor(encodings).float()
        decodings = model._model.decoder(encodings).detach().numpy()
        plt.subplot(6, 6, i + 1)
        plt.imshow(librosa.power_to_db(decodings[0, :, :]), origin="lower", cmap="viridis")
        plt.xticks(())
        plt.yticks(())
    n_cluster = len(cluster_centers)

    if not extra:
        extra= ""
    else:
        pass
    if save and root is not None:
        plt.savefig(f"{root}/Centroids_plot_{n_cluster}_{extra}.pdf", format="pdf")
        plt.show()
    else:
        plt.show()
        print("Ploted!")


def num_rows_cols(num_elements):
    num_rows = int(np.sqrt(num_elements))
    num_cols = (num_elements + num_rows - 1) // num_rows
    return num_rows, num_cols


def get_row_col(pos, cols):
    row = pos // cols
    col = pos % cols
    return row, col


class ClusteringResults:
    def __init__(self, model, df, y_label="hour", hist_library="plt"):
        self._labels_cluster = None
        self._n_labels = None
        self._label = y_label
        self._model = model
        self._n_clusters = len(set(model.labels_))
        self._y = df
        self._n_labels = set(list(self._y.loc[:, self._label]))

    def one_cluster_eval(self, cluster):
        index = np.where(self._model.labels_ == cluster)
        index = list(index[0])
        self._labels_cluster = self._y.loc[index, self._label]
        return list(self._labels_cluster)

    def tagger(self, samples):
        labels = []
        self._y["cluster"] = self._model.labels_
        for cluster in range(self._n_clusters):
            index = np.where(self._model.labels_ == cluster)
            index = index[0]
            labels.append(samples[index])
        return labels

    def joyplot(self, joy_vars=None):
        if joy_vars is None:
            joy_vars = ["hour", "location"]
        size_x = 8
        size_y = 6
        labels_cluster = []
        df = pd.DataFrame()
        for cluster in range(self._n_clusters):
            df = pd.DataFrame()
            for i, label in enumerate(joy_vars):
                labels_cluster.append(self.tagger(np.asarray(self._y[label])))
                df[label] = labels_cluster[i][cluster]

            if (self._label == "location"):
                joypy.joyplot(df, by="location", column="hour", range_style='own',
                              grid="y", hist=False, linewidth=1, legend=False, figsize=(size_x, size_y),
                              title=f"Cluster {cluster} \nLabels distribution along recorders using recorders as rows",
                              colormap=cm.autumn_r, fade=False)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.show()

    def histograms(self, hist_library="plt", extra=None, root=None, save=True):
        bins = list(self._n_labels)
        num_rows, num_cols = num_rows_cols(self._n_clusters)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 14))
        if not extra:
            extra = ""
        else:
            pass
        if self._n_clusters <= 3:
            axes = np.expand_dims(axes, 0)
            fig.set_figheight(6)
            fig.set_figwidth(12)
            if self._n_clusters == 1:
                axes = np.expand_dims(axes, 0)
            else:
                pass
        else:
            pass
        for hist in range(self._n_clusters):
            aux = self.one_cluster_eval(hist)
            ax_0, ax_1 = get_row_col(hist, num_cols)
            if hist_library == "plt":
                axes[ax_0][ax_1].hist(aux, histtype="bar",
                                      color="paleturquoise", cumulative=False,
                                      edgecolor='black',
                                      linewidth=1.2, bins=bins, stacked=False)
                axes[ax_0][ax_1].set_title(f"Cluster: {hist + 1}", size=16)
            elif hist_library == "sns":
                sns.distplot(aux, bins=np.arange(aux.min(), aux.max() + 1),
                             hist_kws=dict(edgecolor="black", linewidth=1),
                             ax=axes[ax_0, ax_1])
                axes[ax_0][ax_1].set_title(f"Cluster: {hist + 1}", size=16)
            else:
                raise Exception(f"Library {self._hist_library} unused")

            if (root is not None) & (save is True):
                plt.savefig(f"{root}/Histograms_plot_{self._n_clusters}_{extra}.pdf", format="pdf")
            else:
                pass
        plt.show()