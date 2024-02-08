import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, roc_auc_score, roc_curve, \
                            mean_squared_error, mean_squared_log_error, \
                            mean_absolute_error
import torch

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def root_mean_squared_log_error(y_true, y_pred):
    return root_mean_squared_error(np.log1p(y_true), np.log1p(y_pred))

def adj_r2_score(y_true, y_pred, p, n = None):

    n = n or len(y_true)

    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n-1) / (n-p-1)

    return adj_r2

def euclidean_distance(X, Y, axis = 1):
    distance = np.sqrt(np.sum(np.abs(X - Y)**2, axis = axis))
    return distance

def manhattan_distance(X, Y, axis = 1):
    distance = np.sum(np.abs(X - Y), axis = axis)
    return distance

def cosine_similarity(X, Y, axis = 1):    
    X_norm = np.linalg.norm(X, axis = axis, keepdims = True)
    Y_norm = np.linalg.norm(Y, axis = axis, keepdims = True)

    if X.ndim == 1 or Y.ndim == 1:
      XY_dot = np.dot(X, Y.T)
    else:
      XY_dot = np.dot(X, Y.T) if axis == 1 else np.dot(X.T, Y)
      
    similarity = XY_dot/(X_norm * Y_norm.T)    
    return similarity

def cosine_dissimilarity(X, Y, axis = 1):    
    
    return 1 - cosine_similarity(X, Y, axis = axis)

def silhouette_score_(data, labels, distance = 'euclidean', greater_is_better = False):

  if distance == 'euclidean':
    distance_fn = euclidean_distance
  elif distance == 'manhattan':
    distance_fn = manhattan_distance
  elif distance == 'cosine_dissimilarity':
    distance_fn = cosine_dissimilarity

  unique_labels = np.unique(labels)
  idx = np.arange(data.shape[0])

  a = []
  b = []
  score = []
  for n in range(data.shape[0]):

    if any((labels == labels[n]) & (idx != n)) and \
       any(any(labels == label) for label in unique_labels if label != labels[n]):
       
      a.append((distance_fn(data[(labels == labels[n]) & (idx != n), :], data[n:(n+1), :])).mean())    
      b.append(np.min([(distance_fn(data[labels == label, :], data[n:(n+1), :])).mean() for label in unique_labels if label != labels[n]]))
    
      if (b[-1] == 0) & (a[-1] == 0):
        score.append(0)
      elif (b[-1] == 0) & (a[-1] != 0):
        score.append(-1)
      elif (b[-1] != 0) & (a[-1] == 0):
        score.append(1)
      else:
        score.append((b[-1] - a[-1]) / np.max([a[-1], b[-1]]))

  score = np.mean(score) if len(score) > 0 else 0

  return score

def dunn_score(data, labels, distance='euclidean', greater_is_better=False):
    # Mapping the distance function based on the specified metric
    distance_fn = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'cosine_dissimilarity': cosine_dissimilarity
    }.get(distance, euclidean_distance)  # Default to euclidean_distance if not matched
    
    unique_labels = np.unique(labels)
    
    max_intra_cluster_distances = []
    min_inter_cluster_distances = []
    
    # Calculate the maximum intra-cluster distance
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        if cluster_indices.size > 1:  # Ensure cluster has more than one member
            intra_distances = distance_fn(data[cluster_indices, :], data[cluster_indices, :][:, None], axis=2).max()
            max_intra_cluster_distances.append(intra_distances)
     
    # Calculate the minimum inter-cluster distance
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)): 
          indices_i = np.where(labels == unique_labels[i])[0]
          indices_j = np.where(labels == unique_labels[j])[0]
          inter_distances = distance_fn(data[indices_i, :], data[indices_j, :][:, None], axis=2).min()
          min_inter_cluster_distances.append(inter_distances)
    
    if not max_intra_cluster_distances or not min_inter_cluster_distances:
        return 0  # Return 0 if unable to calculate due to clustering issues
    
    max_intra_cluster_distance = max(max_intra_cluster_distances)
    min_inter_cluster_distance = min(min_inter_cluster_distances)
    
    score = min_inter_cluster_distance / max_intra_cluster_distance if max_intra_cluster_distance > 0 else 0
    
    return score


def within_cluster_sum_of_squares(data, labels, distance = None):
    
  unique_labels = np.unique(labels)
  
  wcss = 0
  for i in range(len(unique_labels)):
      
      wcss += np.sum(euclidean_distance(data[np.where(labels == unique_labels[i])[0], :], 
                                 data[np.where(labels == unique_labels[i])[0], :].mean(axis = 0),
                                 axis = 1)**2)

  return wcss

def calculate_cluster_scores(data, labels, distance = 'euclidean', scores = None):
  
  if scores is None:
    scores = {'inertia': within_cluster_sum_of_squares,
              'silhouette': silhouette_score_,
              'dunn': dunn_score}
  
  results = {}
  for name, func in scores.items():
      results[name] = func(data = data, labels = labels, distance = distance)

  return results
  
def calculate_scores(y_true, y_pred, scores = None):
  
  if scores is None:
    scores = {'accuracy': accuracy_score,
              'precision': precision_score,
              'recall': recall_score,
              'f1': f1_score,
              'roc_auc': roc_auc_score}

  results = {}
  for name, func in scores.items():
    results[name] = func(y_true, y_pred)

  return results

def evaluate_model(model, X, y, scores = None, model_name = None):

  y_pred = model.predict(X)

  results = calculate_scores(y, y_pred, scores = scores)

  # results['model'] = model_name if model_name is not None else model.__class__.__name__
  
  return results

def get_classification_results(model, X, y, scores = None):
  
  # Labels
  y_pred = model.predict(X)
  # Probabilities
  y_proba = model.predict_proba(X)[:,1]
  ## metrics
  metrics = evaluate_classifier(model, X, y, scores = scores)
  ## ROC curve
  fpr, tpr, _ = roc_curve(y, y_proba, drop_intermediate = False)
  
  results = {'pred': y_pred,
             'proba': y_proba,
             'accuracy': metrics['accuracy'],
             'precision': metrics['precision'],
             'f1': metrics['f1'],
             'roc_auc': metrics['roc_auc'],
             'fpr': fpr,
             'tpr': tpr}

  return results

def plot_roc(fpr, tpr,
             ax = None,
             lw = 1,
             include_diag = True):

  if ax is None:
    fig, ax = plt.subplots(figsize = (5, 5))

  if include_diag:
    ax.plot([0, 1], [0, 1], '--k', lw = 1., alpha = 0.5)

  ax.plot(fpr, tpr, lw = lw)
  ax.grid(True)
  ax.set_xlim([-0.01, 1.01])
  ax.set_ylim([-0.01, 1.01])

class RMSELoss(torch.nn.modules.loss._Loss):
  def __init__(self, reduction = 'mean'):
    super(RMSELoss, self).__init__()

    self.reduction = reduction
    self.name = 'RMSELoss'

  def forward(self, input, target):

    loss = torch.nn.MSLELoss(reduction = self.reduction)(input, target).sqrt()

    return loss

class RMSLELoss(torch.nn.modules.loss._Loss):
  def __init__(self, reduction = 'mean'):
    super(RMSLELoss, self).__init__()

    self.reduction = reduction
    self.name = 'RMSLELoss'

  def forward(self, input, target):

    if (input < 0).any() & ~(target < 0).any():
      raise ValueError("prediction has negative values.")
    elif ~(input < 0).any() & (target < 0).any():
      raise ValueError("target has negative values.")
    elif (input < 0).any() & (target < 0).any():
      raise ValueError("both the prediction and target have negative values.")
    
    loss = torch.nn.MSELoss(reduction = self.reduction)(torch.log1p(input), torch.log1p(target)).sqrt()

    return loss  
