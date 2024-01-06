from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
import numpy as np

def log_transform(X, y = None, small_constant = 1e-10):
  X_t = np.log(X + np.max([small_constant, X.min()]))    
  return X_t

def sqrt_transform(X, y = None, small_constant = 1e-10):
  X_t = np.sqrt(X + np.max([small_constant, X.min()]))    
  return X_t
