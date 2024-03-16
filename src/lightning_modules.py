import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd

## Pytorch Lightning module for classification networkds
class LitClassifier(pl.LightningModule):
    def __init__(self, 
                 model, criterion, optimizer,
                 regularization = None,
                 scores = None, display_score = None,
                 proba_threshold = 0.5,
                 track_performance = False, 
                 track_parameters = False,
                 accelerator = 'cpu', devices = 1):
        
        super(LitClassifier, self).__init__()
        
        self.accelerator = accelerator
        self.devices = devices

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.regularization = regularization
        self.scores = scores
        self.display_score = display_score
        
        self.proba_threshold = proba_threshold

        self.track_performance = track_performance        
        self.track_parameters = track_parameters 

        self.step_loss_train = [ ]
        self.step_loss_val = [ ]
        self.step_loss_test = [ ]

        self.train_history = {'step': [ ]}
        self.train_history['epoch'] = [ ]

        self.train_history['step_loss'] = [ ] 
        self.train_history['epoch_loss'] = [ ] 

        for name, _ in self.model.named_parameters():
            # self.train_history[f"step_{name}"] = [ ]
            self.train_history[f"epoch_{name}"] = [ ]
        
        self.val_history = {'epoch': [ ]}
        self.val_history['epoch_loss'] = [ ] 

        for score in self.scores:
            self.train_history[f"epoch_{score}"] = [ ]
            self.val_history[f"epoch_{score}"] = [ ]

        self.output_train = [ ]
        self.output_val = [ ]
        self.output_test = [ ]

        self.output_pred_train = [ ]
        self.output_pred_val = [ ]
        self.output_pred_test = [ ]

        self.train_scores = None
        self.val_scores = None
        self.test_scores = None
        self.prediction_results = None

    def forward(self, input):
        return self.model(input)
    
    ## Training
    def training_step(self, batch, batch_idx):
        
        input, output = batch

        output_pred = self(input)
        
        self.output_train.append(output)
        self.output_pred_train.append(output_pred)

        loss = self.criterion(output_pred.squeeze(), output.squeeze())
        
        if self.regularization is not None:
            l1_penalty = torch.tensor(0.).to(input.device)
            l2_penalty = torch.tensor(0.).to(input.device)
            
            for param in self.model.parameters():
                if 'l1' in self.regularization:
                    l1_penalty += param.abs().sum()
                if 'l2' in self.regularization:
                    l2_penalty += param.pow(2).sum()
            
            loss += self.regularization.get('l1', 0.) * l1_penalty
            loss += self.regularization.get('l2', 0.) * l2_penalty

        self.step_loss_train.append(loss.mean().item())

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):

        self.train_history['step'].append(self.global_step) 

        self.train_history['step_loss'].append(self.step_loss_train[-1])
        
        # if self.track_parameters:
        #     for name, param in self.model.named_parameters():
        #         self.train_history[f"step_{name}"].append(param.clone().detach().cpu())
            
    def on_train_epoch_end(self):
        
        self.train_history['epoch'].append(self.current_epoch)

        epoch_loss_train = np.mean(self.step_loss_train)

        self.log('train_loss', epoch_loss_train, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        
        if self.track_performance:
            self.train_history['epoch_loss'].append(epoch_loss_train)
        
            if self.scores is not None:
                output = torch.concat(self.output_train, dim = 0).detach().cpu().squeeze()
                output_pred = torch.concat(self.output_pred_train, dim = 0).detach().cpu().squeeze()
                
                if self.proba_threshold is not None:
                    output_pred = output_pred >= self.proba_threshold

                scores_ = calculate_scores(output, output_pred, scores = self.scores)

                self.train_scores = scores_

                for key, value in scores_.items():

                    if self.display_score is None:
                        prog_bar = False
                    else:
                        prog_bar = key in self.display_score
                        
                    self.log(f"train_{key}", value, on_epoch = True, prog_bar = prog_bar, logger = True)

                    self.train_history[f"epoch_{key}"].append(value)

        if self.track_parameters:
            for name, param in self.model.named_parameters():
                self.train_history[f"epoch_{name}"].append(param.clone().detach().cpu())

        self.step_loss_train.clear()
        self.output_train.clear()
        self.output_pred_train.clear()

    ## Validation
    def validation_step(self, batch, batch_idx):
        
        input, output = batch
        
        output_pred = self(input)

        self.output_val.append(output)
        self.output_pred_val.append(output_pred)

        loss = self.criterion(output_pred.squeeze(), output.squeeze())
        
        self.step_loss_val.append(loss.mean().item())

        return loss

    def on_validation_epoch_end(self):
        
        self.val_history['epoch'].append(self.current_epoch)

        epoch_loss_val = np.mean(self.step_loss_val)
        
        self.log('val_loss', epoch_loss_val, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        if self.track_performance:
            self.val_history['epoch_loss'].append(epoch_loss_val)

            if self.scores is not None:
                output = torch.concat(self.output_val, dim = 0).detach().cpu().squeeze()
                output_pred = torch.concat(self.output_pred_val, dim = 0).detach().cpu().squeeze()
                
                if self.proba_threshold is not None:
                    output_pred = output_pred >= self.proba_threshold
                
                scores_ = calculate_scores(output, output_pred, scores = self.scores)

                self.val_scores = scores_

                for key, value in scores_.items():

                    if self.display_score is None:
                        prog_bar = False
                    else:
                        prog_bar = key in self.display_score

                    if self.display_score is None:
                        prog_bar = False
                    else:
                        prog_bar = key in self.display_score

                    self.log(f"val_{key}", value, on_epoch = True, prog_bar = prog_bar, logger = True)

                    self.val_history[f"epoch_{key}"].append(value)

        self.step_loss_val.clear()
        self.output_val.clear()
        self.output_pred_val.clear()

    # Test
    def test_step(self, batch, batch_idx):
        
        input, output = batch
        
        output_pred = self(input)
        
        self.output_test.append(output)
        self.output_pred_test.append(output_pred)
        
        loss = self.criterion(output_pred.squeeze(), output.squeeze())
        
        self.step_loss_test.append(loss.mean().item())

        return loss

    def on_test_epoch_end(self):

        epoch_loss_test = np.mean(self.step_loss_test)
        
        self.log('test_loss', epoch_loss_test, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        if self.scores is not None:
            output = torch.concat(self.output_test, dim = 0).detach().cpu().squeeze()
            output_pred = torch.concat(self.output_pred_test, dim = 0).detach().cpu().squeeze()
            
            if self.proba_threshold is not None:
                output_pred = output_pred >= self.proba_threshold

            scores_ = calculate_scores(output, output_pred, scores = self.scores)
            
            self.test_scores = scores_

            for key, value in scores_.items():

                if self.display_score is None:
                    prog_bar = False
                else:
                    prog_bar = key in self.display_score
                    
                self.log(f"test_{key}", value, on_epoch = True, prog_bar = prog_bar, logger = True)

        self.step_loss_test.clear()
        self.output_test.clear()
        self.output_pred_test.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        
        input, output = batch
        
        output_pred = self(input)
        
        self.target.append(output)
        self.prediction.append(output_pred)

        return output_pred
 
    def on_predict_epoch_start(self):
        self.target = []
        self.prediction = []
        
    def on_predict_epoch_end(self):

        target = np.concatenate(self.target.detach().cpu(), 0)
        prediction = np.concatenate(self.prediction.detach().cpu(), 0)

        results = {'target': target}
        results['prediction'] = prediction

        if self.scores is not None:

          if self.proba_threshold is not None:
              prediction = int(prediction >= self.proba_threshold)
              results['prediction'] = prediction

          scores_ = calculate_scores(target, prediction, scores = self.scores)

          results.update(scores_)

        self.prediction_results = results

    def configure_optimizers(self):

        return self.optimizer        
        output = self.sequential(input).to(self.y_dtype)

        return output

## Pytorch Lightning module for sequence models
class LitSequence(pl.LightningModule):
    def __init__(self,
                 model, 
                 criterion, optimizer,
                 model_type = 'rnn',
                 hiddens = None,
                 regularize = False,
                 scores = None, display_score = None,                 
                 track_performance = False, 
                 track_parameters = False,
                 accelerator = 'cpu', devices = 1):
        
        super(LitSequence, self).__init__()
        
        # Store the arguments as attributes
        locals_ = locals().copy()
        for arg in locals_:
            if arg != 'self':
                setattr(self, arg, locals_[arg])

        self.step_loss_train = [ ]
        self.step_loss_val = [ ]
        self.step_loss_test = [ ]

        self.train_history = {'step': [ ]}
        self.train_history['epoch'] = [ ]

        self.train_history['step_loss'] = [ ] 
        self.train_history['epoch_loss'] = [ ] 

        for name, _ in self.model.named_parameters():
            self.train_history[f"epoch_{name}"] = [ ]
        
        self.val_history = {'epoch': [ ]}
        self.val_history['epoch_loss'] = [ ] 

        if self.scores:  
          for score in self.scores:
              self.train_history[f"epoch_{score}"] = [ ]
              self.val_history[f"epoch_{score}"] = [ ]

        self.output_train = [ ]
        self.output_val = [ ]
        self.output_test = [ ]

        self.output_pred_train = [ ]
        self.output_pred_val = [ ]
        self.output_pred_test = [ ]

        self.train_scores = None
        self.val_scores = None
        self.test_scores = None
        self.prediction_results = None

    def forward(self, input, hiddens = None):
        
        if self.model_type == 'rnn':
          return self.model(input, hiddens)
        else:
          return self.model(input)

    def configure_optimizers(self):
        return self.optimizer
        
    ## Training
    def training_step(self, batch, batch_idx):
        
        input, output = batch
        
        output_pred = self(input)
          
        if self.model_type == 'rnn':
          output_pred, self.hiddens = self(input, self.hiddens)
        else:
          output_pred = self(input)

        self.output_train.append(output)
        self.output_pred_train.append(output_pred)

        loss = self.criterion(output_pred.squeeze(), output.squeeze())
        
        if self.regularize:
          loss += self.regularize()

        self.step_loss_train.append(loss.mean().item())

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):

        self.train_history['step'].append(self.global_step) 

        self.train_history['step_loss'].append(self.step_loss_train[-1])
        
    def on_train_epoch_end(self):
        
        self.train_history['epoch'].append(self.current_epoch)
        
        epoch_loss_train = np.mean(self.step_loss_train)

        self.log('train_loss', epoch_loss_train, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        
        if self.track_performance:
          self.train_history['epoch_loss'].append(epoch_loss_train)
          
          if self.scores is not None:
              output = torch.concat(self.output_train, dim = 0).detach().cpu().squeeze()
              output_pred = torch.concat(self.output_pred_train, dim = 0).detach().cpu().squeeze()
              
              scores_ = calculate_scores(output, output_pred, scores = self.scores)

              self.train_scores = scores_

              for key, value in scores_.items():

                  if self.display_score is None:
                    prog_bar = False
                  else:
                    prog_bar = key in self.display_score

                  self.log(f"train_{key}", value, on_epoch = True, prog_bar = prog_bar, logger = True)

                  self.train_history[f"epoch_{key}"].append(value)

        if self.track_parameters:
            for name, param in self.model.named_parameters():
                self.train_history[f"epoch_{name}"].append(param.clone().detach().cpu())

        self.step_loss_train.clear()
        self.output_train.clear()
        self.output_pred_train.clear()
    
    ## Validation
    def validation_step(self, batch, batch_idx):
        
        input, output = batch

        if self.model_type == 'rnn':
          output_pred, self.hiddens = self(input, self.hiddens)
        else:
          output_pred = self(input)

        self.output_val.append(output)
        self.output_pred_val.append(output_pred)

        loss = self.criterion(output_pred.squeeze(), output.squeeze())
        
        self.step_loss_val.append(loss.mean().item())

        return loss

    def on_validation_epoch_end(self):
        
        self.val_history['epoch'].append(self.current_epoch)

        epoch_loss_val = np.mean(self.step_loss_val)
        
        self.log('val_loss', epoch_loss_val, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        if self.track_performance:
            self.val_history['epoch_loss'].append(epoch_loss_val)

            if self.scores is not None:
                output = torch.concat(self.output_val, dim = 0).detach().cpu().squeeze()
                output_pred = torch.concat(self.output_pred_val, dim = 0).detach().cpu().squeeze()

                scores_ = calculate_scores(output, output_pred, scores = self.scores)

                self.val_scores = scores_

                for key, value in scores_.items():

                    if self.display_score is None:
                        prog_bar = False
                    else:
                        prog_bar = key in self.display_score

                    if self.display_score is None:
                        prog_bar = False
                    else:
                        prog_bar = key in self.display_score

                    self.log(f"val_{key}", value, on_epoch = True, prog_bar = prog_bar, logger = True)

                    self.val_history[f"epoch_{key}"].append(value)

        self.step_loss_val.clear()
        self.output_val.clear()
        self.output_pred_val.clear()

    # Test
    def test_step(self, batch, batch_idx):
        
        input, output = batch

        if self.model_type == 'rnn':
          output_pred, self.hiddens = self(input, self.hiddens)
        else:
          output_pred = self(input)

        self.output_test.append(output)
        self.output_pred_test.append(output_pred)
        
        loss = self.criterion(output_pred.squeeze(), output.squeeze())
        
        self.step_loss_test.append(loss.mean().item())

        return loss
    
    def on_test_epoch_end(self):

        epoch_loss_test = np.mean(self.step_loss_test)
        
        self.log('test_loss', epoch_loss_test, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        if self.scores is not None:
          output = torch.concat(self.output_test, dim = 0).detach().cpu().squeeze()
          output_pred = torch.concat(self.output_pred_test, dim = 0).detach().cpu().squeeze()

          scores_ = calculate_scores(output, output_pred, scores = self.scores)
          
          self.test_scores = scores_

          for key, value in scores_.items():

            if self.display_score is None:
              prog_bar = False
            else:
              prog_bar = key in self.display_score
                
            self.log(f"test_{key}", value, on_epoch = True, prog_bar = prog_bar, logger = True)

        self.step_loss_test.clear()
        self.output_test.clear()
        self.output_pred_test.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        
        input, output = batch

        output_pred = self(input)
        
        if self.model_type == 'rnn':
          output_pred, self.hiddens = self(input, self.hiddens)
        else:
          output_pred = self(input)

        self.target.append(output)
        self.prediction.append(output_pred)

        return output_pred
 
    def on_predict_epoch_start(self):
        self.target = []
        self.prediction = []
        
    def on_predict_epoch_end(self):

        target = np.concatenate(self.target.detach().cpu(), 0)
        prediction = np.concatenate(self.prediction.detach().cpu(), 0)

        results = {'target': target}
        results['prediction'] = prediction
        
        if self.scores is not None:
          
          scores_ = calculate_scores(target, prediction, scores = self.scores)

          results.update(scores_)

        self.prediction_results = results

    def configure_optimizers(self):

        return self.optimizer        
        output = self.sequential(input).to(self.y_dtype)

        return output    
