import torch
import pandas as pd

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
