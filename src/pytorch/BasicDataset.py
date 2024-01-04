import torch
import pandas as pd

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 X, y, 
                 X_dtype = torch.float32, y_dtype = torch.long):

        X = torch.from_numpy(X.replace({False: 0, True: 1}).values if isinstance(X, pd.DataFrame) else X).to(X_dtype).squeeze()
        y = torch.from_numpy(y.replace({False: 0, True: 1}).values if isinstance(y, pd.DataFrame) else y).to(y_dtype).squeeze()
        
        self.X, self.X_dtype = X, X_dtype
        self.y, self.y_dtype = y, y_dtype

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
