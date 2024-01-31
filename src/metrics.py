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
    distance = np.sqrt(np.sum((X - Y)**2, axis = axis))
    return distance

def manhattan_distance(X, Y, axis = 1):
    distance = np.sum(np.abs(X - Y), axis = axis)
    return distance

def cosine_similarity(X, Y, axis = None):
    similarity = np.dot(X, Y) / (np.linalg.norm(X, 2)*np.linalg.norm(Y, 2))
    return similarity

def silhouette_score_(data, labels, metric = 'euclidean', greater_is_better = False):

  if metric == 'euclidean':
    metric_fn = euclidean_distance
  elif metric == 'manhattan':
    metric_fn = manhattan_distance
  elif metric == 'cosine_similarity':
    metric_fn = cosine_similarity
  
  sign_ = (-1)**(1 + ~greater_is_better)

  unique_labels = np.unique(labels)
  idx = np.arange(data.shape[0])

  a = []
  b = []
  score = []
  for n in range(data.shape[0]):
    a.append((sign_ * metric_fn(data[(labels == labels[n]) & (idx != n), :], data[n, :])).mean())
    b.append(np.min([(sign_ * metric_fn(data[labels == label, :], data[n, :])).mean() for label in unique_labels if label != labels[n]]))

    if (b[-1] == 0) & (a[-1] == 0):
      score.append(0)
    elif (b[-1] == 0) & (a[-1] != 0):
      score.append(-1)
    elif (b[-1] != 0) & (a[-1] == 0):
      score.append(1)
    else:
      score.append((b[-1] - a[-1]) / np.max([a[-1], b[-1]]))

  score = np.mean(score)

  return score

def dunn_score(data, labels, metric = 'euclidean', greater_is_better = False):
    
    if metric == 'euclidean':
      metric_fn = euclidean_distance
    elif metric == 'manhattan':
      metric_fn = manhattan_distance
    elif metric == 'cosine_similarity':
      metric_fn = cosine_similarity
    
    sign_ = (-1)**(1 + ~greater_is_better)
    
    unique_labels = np.unique(labels)
    
    max_intra_cluster_distances = []
    min_inter_cluster_distances = []
    
    for i in range(len(unique_labels)):
      ns_i = np.where(labels == unique_labels[i])[0]      
      max_intra_cluster_distances.append(np.max([np.max(sign_ * metric_fn(data[n], data[ns_i], axis = 1)) 
                                                 for n in ns_i]))
      
      for j in range(i+1, len(unique_labels)):
        ns_j = np.where(labels == unique_labels[j])[0] 
        min_inter_cluster_distances.append(np.min([np.min(sign_ * metric_fn(data[n], data[ns_j], axis = 1)) 
                                                   for n in ns_i]))

    max_intra_cluster_distance = np.max(max_intra_cluster_distances)
    min_inter_cluster_distance = np.min(min_inter_cluster_distances)
    
    score = min_inter_cluster_distance / max_intra_cluster_distance

    return score
  
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
