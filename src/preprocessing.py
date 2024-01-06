import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler

def log_transform(X, y = None, small_constant = 1e-10):
  X_t = np.log(X + np.max([small_constant, X.min()]))    
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
      categorical_uniques[col] = unqiues.values
  
  return (df_factorized, categorical_uniques) if return_uniques else df_factorized
