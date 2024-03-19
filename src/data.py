import torch
import pandas as pd

def extract_window_horizon(df,
                           X_names, y_names,
                           window = 1 , horizon = 1, stride = 1,
                           zero_pad_X = False):

  if not isinstance(X_names, list):
    X_names = [X_names]
  if not isinstance(y_names, list):
    y_names = [y_names]

  X = []
  y = []

  t_X = []
  t_y = []

  ar = np.isin(X_names, y_names).any().astype(int)

  for i in range(0, len(df) - window - horizon - ar, stride):

    X_values = df.iloc[i:i+window][X_names].values.reshape(-1, len(X_names))
    y_values = df.iloc[(i+window-1+ar):(i+window-1+ar+horizon)][y_names].values.reshape(-1, len(y_names))

    t_X_values = df.iloc[i:i+window].index.values
    t_y_values = df.iloc[(i+window-1+ar):(i+window-1+ar+horizon)].index.values

    if zero_pad_X:
      pad_size = np.max([window - (i + window), 0])

      X_values = np.pad(X_values,
                        pad_width = ((pad_size, 0), (0, 0)),
                        mode = 'constant',
                        constant_values = (0, 0))

      t_X_values = np.pad(t_X_values,
                          pad_width = ((pad_size, 0)),
                          mode = 'constant',
                          constant_values = (np.nan, 0))

    X.append(X_values)
    y.append(y_values)

    t_X.append(t_X_values)
    t_y.append(t_y_values)

  X = np.stack(X, axis = 0)
  y = np.stack(y, axis = 0)

  t_X = np.array(t_X)
  t_y = np.array(t_y)

  return X, y, t_X, t_y
                               
class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 X, y, 
                 device = 'cpu',
                 X_dtype = torch.float32, y_dtype = torch.long):

        if isinstance(X, pd.DataFrame):
            X = X.copy().values
        elif isinstance(X, pd.Series):
            X = X.copy().values.reshape(-1, 1)
        elif isinstance(X, np.ndarray):
            X = X.copy()
        elif isinstance(X, torch.Tensor):
            X = X.clone().detach()

        if isinstance(y, pd.DataFrame):
            y = y.copy().values
        elif isinstance(y, pd.Series):
            y = y.copy().values.reshape(-1, 1)
        elif isinstance(y, np.ndarray):
            y = y.copy()
        elif isinstance(y, torch.Tensor):
            y = y.clone().detach()

        X = torch.tensor(X).to(device = device, dtype = X_dtype)
        y = torch.tensor(y).to(device = device, dtype = X_dtype)
        
        self.X, self.X_dtype = X, X_dtype
        self.y, self.y_dtype = y, y_dtype

    def __len__(self):
      return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
