import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.cm as cm
import pandas as pd
import torch
import joypy


def plot_silhouette(X, cluster_labels, n_clusters, silhouette_avg, method=None, extra=None, save=False):
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
    if save:
        plt.savefig(f"temporal/clustering_results/{method}/Silhouette_plot_{n_clusters}_{extra}.pdf", format="pdf")
        plt.show()
    else:
        print("Ploted!")
        pass


def plot_centroids(cluster_centers, model, method, extra="", save=True):
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
    if save == True:
        plt.savefig(f"temporal/clustering_results/{method}/Centroids_plot_{n_cluster}_{extra}.pdf", format="pdf")
        plt.show()
    else:
        print("Ploted!")
        pass


def num_rows_cols(num_elements):
    num_rows = int(np.sqrt(num_elements))
    num_cols = (num_elements + num_rows - 1) // num_rows
    return (num_rows, num_cols)


def get_row_col(pos, cols):
    row = pos // cols
    col = pos % cols
    return row, col


class Clustering_Results:
    def __init__(self, model, y, y_label="hour", hist_library="plt"):
        self._labels_cluster = None
        self._n_labels = None
        self._label = y_label
        self._model = model
        self._n_clusters = len(set(model.labels_))
        self.y = y
        self._y = self.converter(y[self._label])
        self._n_labels = set(self._y)

    def converter(self, var):
        aux = []
        for i in range(len(var)):
            aux.append(var[i].item())
        return np.array(aux)

    def one_cluster_eval(self, cluster):
        index = np.where(self._model.labels_ == cluster)
        index = list(index[0])
        self._labels_cluster = self._y[index]
        return self._labels_cluster

    def tagger(self, samples):
        labels = []
        labels_all_clusters = []
        joy_vars = ["hour", "recorder"]
        for cluster in range(self._n_clusters):
            y_aux = []
            labels_cluster = []
            for i, label in enumerate(joy_vars):
                y_aux.append(self.converter(self.y[label]))
                index = np.where(self._model.labels_ == cluster)
                index = list(index[0])
            labels.append(samples[index])
        # return labels

    def joyplot(self):
        labels_all_clusters = []
        size_x = 8
        size_y = 6
        joy_vars = ["hour", "recorder"]
        for cluster in range(self._n_clusters):
            y_aux = []
            labels_cluster = []
            for i, label in enumerate(joy_vars):
                y_aux.append(self.converter(self.y[label]))
                index = np.where(self._model.labels_ == cluster)
                index = list(index[0])
                labels_cluster.append(y_aux[i][index])
            df = pd.DataFrame({'recorder': labels_cluster[0], "hour": labels_cluster[1]})
            if (self._label == "hour"):
                joypy.joyplot(df, by="hour", column="recorder", range_style='own',
                              grid="y", hist=False, linewidth=1, legend=False, figsize=(size_x, size_y),
                              title=f"Cluster {cluster} \nLabels distribution along recorders using recorders as rows",
                              colormap=cm.autumn_r, fade=False)

            elif (self._label == "recorder"):
                joypy.joyplot(df, by="recorder", column="hour", range_style='own',
                              grid="y", hist=False, linewidth=1, legend=False, figsize=(size_x, size_y),
                              title=f"Cluster {cluster} \nLabels distribution along recorders using hours as rows",
                              colormap=cm.autumn_r)

            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            labels_all_clusters.append(index)
            plt.show()
        #             print(len(labels_cluster))
        #             print(labels_cluster[1].shape)
        #             print(labels_cluster[0:10])
        #             print(index[0:20])

        return labels_all_clusters

    def histograms(self, hist_library="plt", root=None, save=True):
        bins = list(self._n_labels)
        print(bins)
        num_rows, num_cols = num_rows_cols(self._n_clusters)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 14))
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
                axes[ax_0][ax_1].set_title(f"Cluster: {hist}", size=16)
            elif hist_library == "sns":
                sns.distplot(aux, bins=np.arange(aux.min(), aux.max() + 1),
                             hist_kws=dict(edgecolor="black", linewidth=1),
                             ax=axes[ax_0, ax_1])
                axes[ax_0][ax_1].set_title(f"Cluster: {hist}", size=16)
            else:
                raise Exception(f"Library {self._hist_library} unused")

            if (root is not None) & (save is True):
                plt.savefig(f"{root}/Histograms_plot_{self._n_clusters}.pdf", format="pdf")
            else:
                pass
        plt.show()
