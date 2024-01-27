import pandas as pd
import numpy as np 

from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from scipy.stats import skew, entropy

def chi2_test(df, 
              categorical_features, target, 
              significance_level = 0.05, only_return_significant = False):

  df = df.copy()

  label_encoder = LabelEncoder()

  for col in categorical_features:
    if df[col].dtype == 'object':
      df[col] = label_encoder.fit_transform(df[col])

  chi2_scores, p_values = chi2(df[categorical_features].squeeze(), df[target].squeeze())

  results = pd.DataFrame({      
                          'Chi2 Score': chi2_scores,
                          'P-value': p_values
                          }, index = categorical_features)

  if only_return_significant:
    results = results.loc[results['P-value'] <= significance_level]

  return results

def anova_f_test(df, 
                 numeric_features, target, 
                 quantile_normal_transform = False,
                 standardize = False,
                 significance_level = 0.05, only_return_significant = False):

  X = df[numeric_features]

  if quantile_normal_transform:
    quantile_transform = QuantileTransformer(output_distribution = 'normal')
    X = pd.DataFrame(quantile_transform.fit_transform(X),
                     columns = numeric_features)
  if standardize:
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X),
                     columns = numeric_features)
    
  f_scores, p_values = f_classif(X.squeeze(), df[target].squeeze())
    
  results = pd.DataFrame({      
      'F Score': f_scores,
      'P-value': p_values
  }, index = numeric_features)

  if only_return_significant:
    results = results.loc[results['P-value'] <= significance_level]

  return results
                   
def skew_test(df, numeric_features):
  skew_scores = skew(df[numeric_features], axis = 0)

  results = pd.DataFrame({'Skew': skew_scores}, index = numeric_features)

  return results

def shannon(data):
  
  _, counts = np.unique(data, return_counts = True)
  proba = counts / counts.sum()
  
  return entropy(proba, base = 2)

def gini(data):

  _, counts = np.unique(data, return_counts = True)
  proba = counts / counts.sum()

  proba = [count/sum(counts) for count in counts]
  return 1 - sum(p**2 for p in proba)

def impurity_score(data, method = 'shannon'):

  if method == 'shannon':
    return shannon(data)
  elif method == 'gini':
    return gini(data)
  else:
    raise ValueError(f"method ({method}) must be 'shannon' or 'gini'.")

def info_gain(split_feature, target, method = 'shannon'):

  info_before = impurity_score(target, method)

  values, counts = np.unique(split_feature, return_counts = True)

  info_after = 0
  for value, count in zip(values, counts):
    subset = target[split_feature == value]
    info_after += impurity_score(subset, method) * count / len(target)

  return info_before - info_after
