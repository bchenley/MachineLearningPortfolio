from data import transform_data, inverse_transform_data

def predict(model, X, 
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
