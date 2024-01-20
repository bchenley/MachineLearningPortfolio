import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, roc_auc_score, roc_curve, \
                            mean_squared_error, mean_squared_log_error, \
                            mean_absolute_error

def root_mean_squared_error(y_true, y_pred):
    return torch.sqrt(mean_squared_error(y_true, y_pred))
  
def root_mean_squared_log_error(y_true, y_pred):
    return torch.sqrt(mean_squared_log_error(y_true, y_pred))
  
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

def evaluate_classifier(model, X, y, scores = None, model_name = None):

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

class Criterion():

  def __init__(self, name='mse', dims=None):

    self.name = name
    self.dims = dims

  def __call__(self, y_pred, y_true, num_params = None):
    
    if self.name == 'mae':
        # Mean Absolute Error (L1 loss)
        if self.dims is not None: criterion = (y_true - y_pred).abs().nanmean(dim = self.dims)
        else: criterion = (y_true - y_pred).abs()
    elif self.name == 'mse':
        # Mean Squared Error
        if self.dims is not None: criterion = (y_true - y_pred).pow(2).nanmean(dim = self.dims)
        else: criterion = (y_true - y_pred).pow(2)
    elif self.name == 'mase':
        # Mean Absolute Scaled Error
        if self.dims is not None: criterion = (y_true - y_pred).abs().nanmean(dim=self.dims) / (y_true.diff(n=1, dim=self.dims).abs().nanmean(dim=self.dims))
        else: criterion = (y_true - y_pred).abs() / y_true.diff(n=1, dim=self.dims).abs()
    elif self.name == 'rmse':
        # Root Mean Squared Error
        if self.dims is not None: criterion = (y_true - y_pred).pow(2).nanmean(dim = self.dims).sqrt()
        else: criterion = (y_true - y_pred).pow(2).sqrt()
    elif self.name == 'nmse':
      # Normalized Mean Squared Error
      if self.dims is not None: criterion = (y_true - y_pred).pow(2).nansum(dim = self.dims) / y_true.pow(2).nansum(dim=self.dims)
      else: criterion = (y_true - y_pred).abs() / y_true.pow(2)
    elif self.name == 'msle':
      # Root mean squared logarithmic error
      if self.dims is not None: criterion = ((y_true+1).log() - (y_pred+1).log()).pow(2).nanmean(dim = self.dims).sqrt()                                            
      else: criterion = ((y_true+1).log() - (y_pred+1).log()).pow(2).sqrt()
    elif self.name == 'mape':
        # Mean Absolute Percentage Error
        if self.dims is not None: criterion = (((y_true - y_pred) / y_true).abs() * 100).nanmean(dim = self.dims)
        else: criterion = (((y_true - y_pred) / y_true).abs() * 100)    
    elif self.name == 'fb':
        # Fractional Bias
        if self.dims is not None: criterion = (y_pred.nansum(dim=self.dims) - y_true.nansum(dim=self.dims)) / y_true.nansum(dim=self.dims) * 100
        else: criterion = 2*(y_pred - y_true)/(y_pred + y_true) # (y_pred - y_true) / y_true * 100
    elif self.name == 'bic':
      N = torch.tensor(y_true.shape[0] if y_true.ndim == 2 else y_true.shape[1]).to(y_true)
      
      error_var = torch.var(y_true - y_pred)      
      criterion = N*torch.log(error_var) + num_params*torch.log(N)
    elif self.name == 'r2':
        if self.dims is not None: 
          if self.dims == (1):
            y_true_mean = y_true.mean(dim = self.dims).unsqueeze(1).repeat(1, y_true.shape[1], 1)
          else:
            y_true_mean = y_true.mean(dim = self.dims)
          criterion = 1 - (y_true - y_pred).pow(2).sum(dim = self.dims)/(y_true - y_true_mean).pow(2).sum(dim = self.dims)
        else: criterion = 1 - (y_true - y_pred).pow(2)/(y_true - y_true.mean(dim = self.dims)).pow(2)
        
    return criterion
