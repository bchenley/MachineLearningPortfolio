import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd

def torch_train(model, criterion, optimizer,
                dl_train,
                dl_val = None,
                epochs = 10,
                hiddens = False):

  criterion_name = str(criterion).split('()')[0]

  train_loss_history = []
  val_loss_history = []

  for epoch in range(epochs):

    train_loss = []
    for X_train, y_train in dl_train:

      if hiddens is False:

        y_train_pred = model(X_train)

      else:
        if isinstance(hiddens, tuple):
          hiddens = [h[:, :X_train.shape[0]].contiguous() for h in hiddens]
        else:
          hiddens = None

        y_train_pred, hiddens = model(X_train, hiddens = hiddens)

      train_loss_batch = criterion(y_train_pred, y_train)

      optimizer.zero_grad()
      train_loss_batch.sum().backward()
      optimizer.step()

      train_loss.append(train_loss_batch.sum().detach().item())

      if hiddens is not None:
        hiddens = [h.detach() for h in hiddens]

    val_loss = []
    if dl_val is not None:
      for X_val, y_val in dl_val:

        if hiddens is False:

          y_val_pred = model(X_val)

        else:

          if isinstance(hiddens, tuple):
            hiddens = [h[:, :X_val.shape[0]].contiguous() for h in hiddens]
          else:
            hiddens = None

          y_val_pred, hiddens = model(X_val, hiddens = hiddens)


        val_loss_batch = criterion(y_val_pred, y_val)

        val_loss.append(val_loss_batch.sum().detach().item())

      if hiddens:
        hiddens = None

    train_loss_history.append(np.mean(train_loss))

    if len(val_loss) > 0:
      val_loss_history.append(np.mean(val_loss))

      print(f"Epoch {epoch+1}/{epochs}: Training {criterion_name} = {train_loss_history[-1].round(4)}" \
              f" Validation {criterion_name} = {val_loss_history[-1].round(4)}")
    else:
      print(f"Epoch {epoch+1}/{epochs}: Training {criterion_name} = {train_loss_history[-1].round(4)}")

  return train_loss_history, val_loss_history
                    
class CustomRegressionTrainer(pl.Trainer):
    def __init__(self, 
                 max_epochs = None):
        
        super().__init__(max_epochs = max_epochs)
        
    def fit_model(self, 
                  pl_model, 
                  train_dataloaders, val_dataloaders = None, 
                  model_to_device = 'cpu'):

        self.fit(model = pl_model, 
                 train_dataloaders = train_dataloaders,
                 val_dataloaders = val_dataloaders)
        
        pl_model.model.to(model_to_device)
        
    def test_model(self, test_dataloaders):
        
        self.test(model = self.model,
                  dataloaders = test_dataloaders)

    def plot_performance(self, 
                         metrics = None,
                         figsize = None,
                         fig_num = None):
        
        # loss
        fig = plt.figure(num = fig_num, figsize = figsize)

        loss_ax = fig.add_subplot(1,1, 1)
        loss_ax.plot(self.model.train_history['epoch'], 
                     self.model.train_history['epoch_loss'], 'k', label = 'Train')
        if len(self.model.val_history['epoch']) > 0:
            loss_ax.plot(self.model.val_history['epoch'], 
                        self.model.val_history['epoch_loss'], 'r', label = 'Val')
        loss_ax.grid()
        loss_ax.set_xlabel('Epochs')
        loss_ax.set_ylabel(self.model.criterion._get_name())
        loss_ax.legend()

        # metrics
        if metrics is not None:
          m = 1
          for metric in metrics:
            m += 1
            metric_ax = fig.add_subplot(m+1, 1, m)

            metric_ax.plot(self.model.train_history['epoch'], 
                            self.model.train_history[f"epoch_{metric}"], 'k', label = 'Train')
            
            if len(self.model.val_history['epoch']) > 0:
                metric_ax.plot(self.model.val_history['epoch'], 
                            self.model.val_history[f"epoch_{metric}"], 'r', label = 'Val')
            metric_ax.grid()
            metric_ax.set_xlabel('Epochs')
            metric_ax.set_ylabel(self.model.criterion._get_name())

            metric_ax.legend()
        
    def plot_params(self,                                           
                    params = 'all',
                    figsize = None, 
                    fig_num = None):

        fig = plt.figure(num = fig_num, figsize = figsize)

        p = 0
        for name, _ in self.model.named_parameters():
            if (params == 'all') | np.isin(name, params):
                p += 1
                param_ax = fig.add_subplot(p,1,p-1)

                param_ax.plot(self.model.train_history['epoch'],
                               self.model.train_history[f"epoch_{name}"])
                param_ax.xlabel('Epochs')
                param_ax.ylabel(name)
                param_ax.grid()

    def summarise_results(self):

        results = pd.DataFrame({key: [value] for key,value in self.model.train_scores.items()},
                               index = ['Train'])
            
        if self.model.val_scores is not None:
            val_results = pd.DataFrame({key: [value] for key,value in self.model.val_scores.items()},
                                       index = ['Validation'])
            results = pd.concat([results, val_results], axis = 0)
            
        if self.model.test_scores is not None:
            test_results = pd.DataFrame({key: [value] for key,value in self.model.test_scores.items()},
                                        index = ['Test'])
            results = pd.concat([results, test_results], axis = 0)

        return results

    def predict_data(self, dataloaders):
        
        prediction_batches = self.predict(model = self.model, 
                                          dataloaders = dataloaders)
        
        predictions = torch.cat(prediction_batches, 0).cpu().numpy().squeeze()

        return predictions
        
class CustomClassificationTrainer(pl.Trainer):
    def __init__(self, 
                 max_epochs = None):
        
        super().__init__(max_epochs = max_epochs)
        
    def fit_model(self, model, train_dataloaders, val_dataloaders = None):

        self.model = model
            
        self.fit(model = self.model, 
                 train_dataloaders = train_dataloaders,
                 val_dataloaders = val_dataloaders)
        
    def test_model(self, test_dataloaders):
        
        self.test(model = self.model,
                  dataloaders = test_dataloaders)

    def plot_performance(self, 
                         metrics = None,
                         figsize = None,
                         fig_num = None):
        
        # loss
        fig = plt.figure(num = fig_num, figsize = figsize)

        loss_ax = fig.add_subplot(1,1, 0)
        loss_ax.plot(self.model.train_history['epoch'], 
                     self.model.train_history['epoch_loss'], 'k', label = 'Train')
        if len(self.model.val_history['epoch']) > 0:
            loss_ax.plot(self.model.val_history['epoch'], 
                        self.model.val_history['epoch_loss'], 'r', label = 'Val')
        loss_ax.grid()
        loss_ax.set_xlabel('Epochs')
        loss_ax.set_ylabel(self.model.criterion._get_name())
        loss_ax.legend()

        # metrics
        if metrics is not None:
            m = 0
            for metric in metrics:
                m += 1
                metric_ax = fig.add_subplot(m+1, 1, m)

                metric_ax.plot(self.model.train_history['epoch'], 
                               self.model.train_history[f"epoch_{metric}"], 'k', label = 'Train')
                
                if len(self.model.val_history['epoch']) > 0:
                    metric_ax.plot(self.model.val_history['epoch'], 
                                self.model.val_history[f"epoch_{metric}"], 'r', label = 'Val')
                metric_ax.grid()
                metric_ax.set_xlabel('Epochs')
                metric_ax.set_ylabel(self.model.criterion._get_name())

            metric_ax.legend()
        
    def plot_params(self,                                           
                    params = 'all',
                    figsize = None, 
                    fig_num = None):

        fig = plt.figure(num = fig_num, figsize = figsize)

        p = 0
        for name, _ in self.model.named_parameters():
            if (params == 'all') | np.isin(name, params):
                p += 1
                param_ax = fig.add_subplot(p,1,p-1)

                param_ax.plot(self.model.train_history['epoch'],
                               self.model.train_history[f"epoch_{name}"])
                param_ax.xlabel('Epochs')
                param_ax.ylabel(name)
                param_ax.grid()

    def summarise_results(self):

        results = pd.DataFrame({key: [value] for key,value in self.model.train_scores.items()},
                               index = ['Train'])
            
        if self.model.val_scores is not None:
            val_results = pd.DataFrame({key: [value] for key,value in self.model.val_scores.items()},
                                       index = ['Validation'])
            results = pd.concat([results, val_results], axis = 0)
            
        if self.model.test_scores is not None:
            test_results = pd.DataFrame({key: [value] for key,value in self.model.test_scores.items()},
                                        index = ['Test'])
            results = pd.concat([results, test_results], axis = 0)

        return results

    def predict_data(self, dataloaders):
        
        prediction_batches = self.predict(model = self.model, 
                                          dataloaders = dataloaders)
        
        predictions = torch.cat(prediction_batches, 0).cpu().numpy().squeeze()

        if self.model.proba_threshold is not None:
            predictions = (predictions >= self.model.proba_threshold).astype(int)

        return predictions
