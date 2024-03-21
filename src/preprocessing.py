import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
import torch 

def transform_data(X, transformer):

  was_tensor, device, dtype = False, None, None
  if isinstance(X, torch.Tensor):
    was_tensor = True
    device, dtype = X.device, X.dtype
    X = X.detach().cpu().numpy()

  if isinstance(transformer, OneHotEncoder):
    X_transformed = transformer.transform(X.reshape(-1, X.shape[-1])).toarray()
  else:
    X_transformed = transformer.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

  if was_tensor:
    X_transformed = torch.tensor(X_transformed).to(device = device, dtype = dtype)
    
  return X_transformed

def inverse_transform_data(X, transformer):

  was_tensor, device, dtype = False, None, None
  if isinstance(X, torch.Tensor):
    was_tensor = True
    X = X.detach().cpu().numpy()

  if isinstance(transformer, OneHotEncoder):
    X_inv = transformer.inverse_transform(X.reshape(-1, X.shape[-1]))
  else:
    X_inv = transformer.inverse_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

  if was_tensor:
    X_inv = torch.tensor(X_inv).to(device = device, dtype = dtype)
    
  return X_inv

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
