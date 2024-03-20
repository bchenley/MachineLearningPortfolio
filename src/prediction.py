from data import transform_data, inverse_transform_data
import torch

def torch_predict(model, X, 
                  hiddens = False,
                  X_transformer = None,
                  y_transformer = None):

  if X_transformer is not None:
    X = transform_data(X, X_transformer)
            
  with torch.no_grad():
    if hiddens is False:
      y_pred = model(X)
    else:
      if isinstance(hiddens, tuple):
        hiddens = [h[:, :X.shape[0]].contiguous() for h in hiddens]
      else:
        hiddens = None
      
      y_pred, hiddens = model(X, hiddens = hiddens)

    if y_transformer is not None:
      y_pred = inverse_transform_data(y_pred, y_transformer)

  return y_pred, hiddens

def torch_forecst(horizon, 
                  model, X, 
                  output_size = 1,
                  hiddens = None,
                  Xy_ar_idx = None,
                  y_dtype = torch.float32,
                  X_transformer = None,
                  y_transformer = None):
  
  if X_transformer is not None:
    X = transform_data(X, X_transformer)

  y_forecast = torch.empty(size = (1, 0, output_size)).to(device = X.device,
                                                           dtype = y_dtype)
  
  with torch.no_grad():
    
    while y_forecast.shape[1] < horizon:

      y_pred, hiddens = torch_predict(model, X, hiddens = hiddens)
      
      y_forecast = torch.cat((y_forecast, y_pred), dim = 1)

      if Xy_ar_idx is not None:
        X_ = torch.zeros(size = (1, y_pred.shape[1], X.shape[-1])).to(X)
        
        X_[:, :, Xy_ar_idx[0]] = y_pred[:, :, Xy_ar_idx[1]]
        
        X = torch.cat((X, X_), dim = 1)

  if y_transformer is not None:
    y_forecast = inverse_transform_data(y_forecast, y_transformer)

  return y_forecast
