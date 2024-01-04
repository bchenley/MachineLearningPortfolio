from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_classifier(y_true, y_pred, scores = None):
  
  if scores is None:
    scores = {'accuracy': accuracy_score,
              'precision': precision_score,
              'recall': recall_score,
              'f1': f1_score,
              'roc_auc': roc_auc_score}

  result = {}
  for name, func in scores.items():
    result[name] = func(y_true, y_pred)

  return result
