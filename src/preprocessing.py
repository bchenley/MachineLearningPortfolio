import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler

def log_transform(X, y = None, small_constant = 1.):  
  X_t = np.log(X - X.min() + small_constant)    
  return X_t

def sqrt_transform(X, y = None, small_constant = 1e-10):
  X_t = np.sqrt(X + np.max([small_constant, X.min()]))    
  return X_t

def factorize(df, return_uniques = False):
  df_factorized = df.copy()

  if return_uniques:
    categorical_uniques = {}
  for col in df_factorized.select_dtypes(include = ['object']).columns:
    codes, unqiues = pd.factorize(df_factorized[col])
    df_factorized[col] = codes
    if return_uniques:
      categorical_uniques[col] = unqiues.tolist()
  
  return (df_factorized, categorical_uniques) if return_uniques else df_factorized

def create_bins(data, method = 'sturges', num_bins = None):

  N = len(data)
  
  min, max = np.min(data), np.max(data)
  iqr = stats.iqr(data) 

  if num_bins is None:
    if method == 'sturges':
      num_bins = int(1 + np.sqrt(N))
    elif method == 'freedman_diaconis':
      num_bins = int((max - min) / (2 * iqr / N**(1/3)))  
    elif method == 'sqrt':
      num_bins = int(np.qrt(N))

  # Calculate the bin edges
  bins = np.linspace(min, max, num_bins + 1)

  # Create the histogram
  data_binned, _ = np.histogram(data, bins = bins)
  
  bin_mapping = np.digitize(data, bins)

  return bin_mapping, bins
