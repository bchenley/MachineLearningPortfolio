import pandas as pd
import numpy as np

from sklearn.cluster import kmeans_plusplus

from metrics import euclidean_distance, manhattan_distance, cosine_similarity

class CustomKMeans():
    def __init__(self,
                 n_clusters = 2,
                 init = 'k-means++',
                 max_iter = 100,
                 metric = 'euclidean',
                 random_state = None):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state

        self.greater_is_better = False
        if self.metric == 'euclidean':
            self.metric_fn = euclidean_distance
        elif self.metric == 'manhattan':
            self.metric_fn = manhattan_distance
        elif self.metric == 'cosine_similarity':
            self.metric_fn = cosine_similarity
            self.greater_is_better = True

        self.cluster_centers_ = None
        self.labels_ = None

    def assign_clusters(self, data):

        distances = []
        for n in range(self.n_clusters):
            distances.append((-1)**(1 + ~self.greater_is_better) * self.metric_fn(data, self.cluster_centers_[n]))

        distances = np.stack(distances, axis = 1)

        labels_ = distances.argmin(axis = 1)

        return labels_

    def update_cluster_centers(self, data):

        cluster_centers = []
        for n in range(self.n_clusters):
            if np.any(self.labels_ == self.labels_[n]):
                cluster_centers.append(data[self.labels_ == self.labels_[n], :].mean(axis = 0))
            else:
                cluster_centers.append(data[:, ((-1)**(1 + ~self.greater_is_better)*self.metric_fn(data, self.cluster_centers_[n])).argmax()])

        cluster_centers = np.stack(cluster_centers, axis = 0)

        return cluster_centers

    def fit(self, data):

        self.n_samples = data.shape[0]

        np.random.seed(self.random_state)

        if self.labels_ is None:
            self.labels_ = np.random.randint(0, self.n_clusters, (self.n_samples,))

        if self.init == 'k-means++':
            self.cluster_centers_, _ = kmeans_plusplus(data, n_clusters = self.n_clusters)
        elif self.init == 'random':
            self.cluster_centers_ = data[np.random.randint(0, data.shape[0], (self.n_clusters, data.shape[1]))]

        new_labels_ = self.assign_clusters(data)

        itr = 0
        while (any(self.labels_ != new_labels_) and (itr <= self.max_iter)):
            self.labels_ = new_labels_
            self.cluster_centers_ = self.update_cluster_centers(data)
            new_labels_ = self.assign_clusters(data)
            itr += 1

    def predict(self, data):
      return self.assign_clusters(data)

    def fit_predict(self, data):

      self.fit(data)

      return self.predict(data)
