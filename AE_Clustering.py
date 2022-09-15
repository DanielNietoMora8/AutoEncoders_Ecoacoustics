from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import torch 

class AE_Clustering:

    def __init__(self, AE_testing, dataset, n_clusters: int = 27):
        self._ae_testing = AE_testing
        self._dataset = dataset
        self._n_clusters = n_clusters

    def labeling(self, label, repetitions: int = 4, axes: int = 0):
        le = preprocessing.LabelEncoder()
        labels = np.array(label)
        labels= np.repeat(label, repetitions, axes)
        le.fit(labels)
        labels = le.transform(labels)
        return labels

    def plot_clusters(self, X_embedded, original_labels, cluster_labels):
        plt.close("all")
        #output.clear()
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(2, 1, 1)
        ax.scatter(X_embedded[:,0 ], X_embedded[:, 1], c=cluster_labels)
        ax = fig.add_subplot(2, 1, 2)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=original_labels)
        plt.show()

    def plot_centroids(self):
        plt.figure(figsize=(18, 18))
        self._ae_testing._model.to("cpu")
        for i, spec in enumerate(self.kmeans.cluster_centers_):
            encodings = spec.reshape(self._encodings_size)
            encodings = torch.tensor(encodings).float()
            decodings = self._ae_testing.model.decoder(encodings).detach().numpy()
            plt.subplot(9, 9, i + 1)
            plt.imshow(decodings[0, :, :], cmap="inferno", interpolation="nearest", vmin=0, vmax=0.02)
            plt.xticks(())
            plt.yticks(())

    def fordward(self):
        for id, item in enumerate(self._dataset):
            self._ae_testing._model.to("cuda")
            originals, reconstructions, encodings, label, loss = self._ae_testing.reconstruct()
            self._encodings_size = encodings[0].shape
            #label = label.to("cpu")
            labels = self.labeling(label, repetitions=4, axes=0)
            self.kmeans = MiniBatchKMeans(n_clusters=self._n_clusters, random_state=0)
            encodings = encodings.to("cpu").detach()
            encodings = encodings.reshape(216,
                                        encodings.shape[1]*encodings.shape[2]*encodings.shape[3])
            self.kmeans = self.kmeans.partial_fit(encodings)
            embedding = self.kmeans.transform(encodings)
            mbk_means_cluster_centers = self.kmeans.cluster_centers_
            # mbk_means_labels = pairwise_distances_argmin(encodings, mbk_means_cluster_centers)
            mbk_means_labels = self.kmeans.predict(encodings)
            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(encodings)
            print(X_embedded.shape)
            self.plot_clusters(X_embedded, mbk_means_labels, labels)
        return self.kmeans