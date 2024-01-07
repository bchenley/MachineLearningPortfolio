import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def evaluate_classifier(y_true, y_pred, scores = None):
  
  if scores is None:
    scores = {'accuracy': accuracy_score,
              'precision': precision_score,
              'recall': recall_score,
              'f1': f1_score,
              'roc_auc': roc_auc_score}

  results = {}
  for name, func in scores.items():
    results[name] = func(y_true, y_pred)

  results = pd.DataFrame(results, index = scores)
  
  return results

def get_classification_results(model, X, y):

  # Labels
  y_pred = model.predict(X)
  # Probabilities
  y_proba = model.predict_proba(X)[:,1]
  ## metrics
  metrics = evaluate_classifier(y, y_pred)
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
