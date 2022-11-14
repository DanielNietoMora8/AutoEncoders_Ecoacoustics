from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import torch
# import umap
import matplotlib.cm as cm
from sklearn import metrics
from sklearn.metrics import silhouette_samples
import pickle as pkl

class AE_Clustering:

    def __init__(self, AE_testing, dataset, n_clusters: int = 27):
        self._ae_testing = AE_testing
        self._dataset = dataset
        self._n_clusters = n_clusters

    def labeling(self, label, repetitions: int = 4, axes: int = 0):
        le = preprocessing.LabelEncoder()
        labels = np.array(label)
        #labels = np.repeat(label, repetitions, axes)
        le.fit(labels)
        labels = le.transform(labels)
        return labels

    def plot_clusters(self, X_embedded, original_labels, cluster_labels):
        plt.close("all")
        #output.clear()
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(2, 1, 1)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=cluster_labels)
        ax = fig.add_subplot(2, 1, 2)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=original_labels)
        plt.show()

    def plot_silhouette(self, X, cluster_labels, n_clusters, silhouette_avg):
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
        print("Ya debio plotear")
        plt.savefig(f"Clustering_Results/Figures/Clustering_plot_{n_clusters}.pdf", format="pdf")
        plt.show()

    def plot_centroids(self):
        plt.figure(figsize=(18, 18))
        self._ae_testing._model.to("cpu")
        for i, spec in enumerate(self.kmeans.cluster_centers_):
            encodings = spec.reshape(self._encodings_size)
            encodings = torch.tensor(encodings).float()
            decodings = self._ae_testing._model.decoder(encodings).detach().numpy()
            plt.subplot(9, 9, i + 1)
            plt.imshow(decodings[0, :, :], origin="lower", cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
            plt.xticks(())
            plt.yticks(())

    def fordward(self, plot_clusters_period=30):
        silhouette_score_TSNE = []
        silhouette_score_UMAP = []

        self.kmeans = MiniBatchKMeans(n_clusters=self._n_clusters, random_state=0)
        for id, item in enumerate(self._dataset):
            if id+1 == 60:
                break
            else:
                pass
            print(f"id: {id+1} of {len(self._dataset)}")
            self._ae_testing._model.to("cuda")
            try:
                originals, reconstructions, encodings, label, loss = self._ae_testing.reconstruct()
            except:
                continue
            self._encodings_size = encodings[0].shape
            labels = self.labeling(label, repetitions=4, axes=0)

            encodings = encodings.to("cpu").detach()
            encodings = encodings.reshape(encodings.shape[0],
                                        encodings.shape[1]*encodings.shape[2]*encodings.shape[3])
            self.kmeans = self.kmeans.partial_fit(encodings)
            # embedding = self.kmeans.transform(encodings)
            # mbk_means_cluster_centers = self.kmeans.cluster_centers_
            # mbk_means_labels = pairwise_distances_argmin(encodings, mbk_means_cluster_centers)
            mbk_means_labels = self.kmeans.predict(encodings)
            X_embedded_TSNE = TSNE(n_components=2, learning_rate='auto',
                                   init='random', random_state=0).fit_transform(encodings)
            # reducer = umap.UMAP()
            # X_embedded_UMAP = reducer.fit_transform(encodings)

            silhouette_score_TSNE.append(metrics.silhouette_score(encodings, mbk_means_labels))
            print(silhouette_score_TSNE[id])
            if (id+1) % plot_clusters_period == 0:
                self.plot_clusters(X_embedded_TSNE, mbk_means_labels, labels)
                # self.plot_clusters(X_embedded_UMAP, mbk_means_labels, labels)
            else:
                pass
            if (id+1) % 59 == 0:
                print("plotting silhouette graph Embedded")
                self.plot_silhouette(encodings, mbk_means_labels, self._n_clusters, silhouette_score_TSNE[id])
            else:
                pass

        with open(f"Clustering_Results/Results/silhouette_n-clusters: {self._n_clusters}_id: {id}", "wb") as file:
            pkl.dump(silhouette_score_TSNE, file)

        return self.kmeans