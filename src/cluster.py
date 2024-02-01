import pandas as pd
import numpy as np

from sklearn.cluster import kmeans_plusplus

from metrics import euclidean_distance, manhattan_distance, cosine_similarity

class CustomKMeans():
    def __init__(self,
                 n_clusters = 2,
                 init = 'k-means++',
                 max_iter = 100,
                 distance = 'euclidean',
                 random_state = None):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.distance = distance
        self.random_state = random_state

        self.greater_is_better = False
        self.distance_fn = None
        if self.distance == 'euclidean':
            self.distance_fn = euclidean_distance
        elif self.distance == 'manhattan':
            self.distance_fn = manhattan_distance
        elif self.distance == 'cosine_similarity':
            self.distance_fn = cosine_similarity
            self.greater_is_better = True

        self.cluster_centers_ = None
        self.labels_ = None

    def assign_clusters(self, data):

        distances = []
        for n in range(self.n_clusters):
            distances.append((-1)**(1 + ~self.greater_is_better) * self.distance_fn(data, self.cluster_centers_[n]))

        distances = np.stack(distances, axis = 1)
        
        labels_ = distances.argmin(axis = 1)
        
        return labels_

    def update_cluster_centers(self, data):

        cluster_centers = []
        for n in range(self.n_clusters):
            if np.any(self.labels_ == self.labels_[n]):
                cluster_centers.append(data[self.labels_ == self.labels_[n], :].mean(axis = 0))
            else:
                cluster_centers.append(data[:, ((-1)**(1 + ~self.greater_is_better)*self.distance_fn(data, self.cluster_centers_[n])).argmax()])

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
            
        self.inertia_ = within_cluster_sum_of_squares(data, self.labels_, self.distance)
        # self.dunn_ = dunn_score(data, self.labels_, self.distance)
        # self.silhouette_ = silhouette_score(data, self.labels_, self.distance)
    
    def predict(self, data):
      return self.assign_clusters(data)

    def fit_predict(self, data):

      self.fit(data)

      return self.predict(data)
    
    def get_params(self, deep = False):

      return {'n_clusters': self.n_clusters,
              'init': self.init,
              'max_iter': self.max_iter,
              'distance': self.distance,
              'random_state': self.random_state}

    def set_params(self, params):
      
      for param, value in params.items():
        setattr(self, param, value)
      
      return self

    def elbow_test(self,
                   train_data, val_data = None,
                   Ks = [k for k in range(2, 6)], 
                   distances = ['euclidean'],
                   scores = ['inertia']):
      
      val_data = val_data or train_data

      init_params = self.get_params()

      results = {'n_clusters': [], 'distance': []}

      for score in scores: 
        results[score] = []

      for distance in distances:
        for n_clusters in Ks:

          self.set_params({'n_clusters': n_clusters,
                           'distance': distance})
          
          self.fit(train_data)

          labels = self.predict(val_data)

          scores_ = calculate_cluster_scores(data = val_data, labels = labels, distance = distance) 

          results['n_clusters'].append(n_clusters)
          results['distance'].append(distance)
          
          for score in scores:            
            results[score].append(scores_[score])

      # Set params back to init values
      self.set_params(init_params)

      return results
