import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd

## 1D CNN
class CNN1D(torch.nn.Module):

    def __init__(self, 
                 in_channels, out_channels, input_len = 1,
                 causal_pad = False,
                 kernel_size = [(1,)], kernel_stride = [(1,)], padding = [(0,)], 
                 dilation = [(1,)], groups = [1], bias = [False], 
                 pool_type = [None], pool_size = [(2,)], pool_stride = [(1,)],
                 activation = ['identity'],
                 degree = [2], coef_init = [None], coef_train = [True], coef_reg = [[0.001, 1]] , zero_order = [False],
                 batch_norm = False, batch_norm_learn = False,
                 dropout_p = [0.],
                 device = None, dtype = None):

        super(CNN1D, self).__init__()
        
        # Store the arguments as attributes
        locals_ = locals().copy()
        for arg in locals_:
            if arg != 'self':
                setattr(self, arg, locals_[arg])

        self.num_layers = len(self.out_channels)

        # Replicate single-value parameters to match the number of layers
        if len(self.kernel_size) == 1: self.kernel_size = self.kernel_size * self.num_layers
        if len(self.kernel_stride) == 1: self.kernel_stride = self.kernel_stride * self.num_layers
        if len(self.padding) == 1: self.padding = self.padding * self.num_layers
        if len(self.dilation) == 1: self.dilation = self.dilation * self.num_layers
        if len(self.groups) == 1: self.groups = self.groups * self.num_layers
        if len(self.bias) == 1: self.bias = self.bias * self.num_layers
        if len(self.activation) == 1: self.activation = self.activation * self.num_layers
        if len(self.degree) == 1: self.degree = self.degree * self.num_layers
        if len(self.coef_init) == 1: self.coef_init = self.coef_init * self.num_layers
        if len(self.coef_train) == 1: self.coef_train = self.coef_train * self.num_layers
        if len(self.coef_reg) == 1: self.coef_reg = self.coef_reg * self.num_layers
        if len(self.zero_order) == 1: self.zero_order = self.zero_order * self.num_layers
        if len(self.pool_type) == 1: self.pool_type = self.pool_type * self.num_layers
        if len(self.pool_size) == 1: self.pool_size = self.pool_size * self.num_layers
        if len(self.pool_stride) == 1: self.pool_stride = self.pool_stride * self.num_layers
        if len(self.dropout_p) == 1: self.dropout_p = self.dropout_p * self.num_layers
            
        # Create the CNN layers
        self.cnn = torch.nn.ModuleList()  
        for i in range(self.num_layers):
            self.cnn.append(torch.nn.Sequential())
            
            # Determine the input channels for the current layer
            in_channels_i = self.in_channels if i == 0 else self.out_channels[i - 1]

            # 1) Add Conv1d layer to the current layer
            self.cnn[-1].append(torch.nn.Conv1d(in_channels = in_channels_i,
                                                out_channels = self.out_channels[i],
                                                kernel_size = self.kernel_size[i],
                                                stride = self.kernel_stride[i],
                                                padding = self.padding[i],
                                                dilation = self.dilation[i],
                                                groups = self.groups[i],
                                                bias = self.bias[i],
                                                device = self.device, dtype = self.dtype))
            
            # 2) Add batch normalization layer if specified
            if self.batch_norm:
                batch_norm_i = torch.nn.BatchNorm1d(self.out_channels[i], 
                                                    affine = self.batch_norm_learn,
                                                    device = self.device, dtype = self.dtype)
            else:
                batch_norm_i = torch.nn.Identity()
            
            self.cnn[-1].append(batch_norm_i)

            # 3) Add activation
            if (self.activation[i] == 'identity') or (self.activation[i] is None):
              activation_fn = torch.nn.Identity()
            elif self.activation[i] == 'relu':
              activation_fn = torch.nn.ReLU()
            elif self.activation[i] == 'polynomial':
              activation_fn = Polynomial(in_features = self.out_channels[i],
                                         degree = self.degree[i],
                                         coef_init = self.coef_init[i],
                                         coef_train = self.coef_train[i],
                                         coef_reg = self.coef_reg[i],
                                         zero_order = self.zero_order[i],
                                         device = self.device,
                                         dtype = self.dtype)
            
            self.cnn[-1].append(activation_fn)
            
            # 4) Add pooling layer if specified
            if self.pool_type[i] is None:
                pool_i = torch.nn.Identity()
            if self.pool_type[i] == 'max':
                pool_i = torch.nn.MaxPool1d(kernel_size = self.pool_size[i],
                                            stride = self.pool_stride[i])
            elif self.pool_type[i] == 'avg':
                pool_i = torch.nn.AvgPool1d(kernel_size = self.pool_size[i],
                                            stride = self.pool_stride[i])     

            self.cnn[-1].append(pool_i)

            # 5) Dropout
            self.cnn[-1].append(torch.nn.Dropout(self.dropout_p[i]))
                     
        # Determine the number of output features after passing through the layers
        with torch.no_grad():             
          X = torch.zeros((2, self.input_len, in_channels)).to(device = self.device, dtype = self.dtype)
          self.output_len = self.forward(X).shape[1]
    
    def forward(self, input):

        output = input.clone()
        for i in range(self.num_layers):   
          # Apply padding to the input tensor if causal_pad is True
          input_i = torch.nn.functional.pad(output, (0, 0, self.kernel_size[i][0] - 1, 0, 0, 0)) if self.causal_pad else output
          # Apply the current CNN layer to the input tensor
          output = self.cnn[i][0](input_i.permute(0, 2, 1)).permute(0, 2, 1) 
          # Apply batch normalization
          output = self.cnn[i][1](output.permute(0, 2, 1)).permute(0, 2, 1)
          # Apply activation
          output = self.cnn[i][2](output)
          # Apply pooling
          output = self.cnn[i][3](output.permute(0, 2, 1)).permute(0, 2, 1)
          # Apply dropout
          output = self.cnn[i][4](output)
          
        # Transpose back the output tensor to the original shape
        output = output

        return output
        
## Custom RNN network
class CustomRNN(torch.nn.Module):
    def __init__(self,
                 input_size, output_size = 1, input_len = 1, output_len = 1,
                 rnn_type = 'lstm',
                 stateful = False,
                 hidden_size = 32, 
                 rnn_num_layers = 1, 
                 rnn_bias = True, 
                 rnn_batch_first = True, 
                 rnn_dropout = 0, 
                 rnn_bidirectional = False,
                 lstm_proj_size = 0, 
                 rnn_weight_reg = [0.001, 1],
                 output_out_features = [1], 
                 output_bias = [True], 
                 output_weight_reg = [0.001, 1], 
                 output_activation = ['relu'],
                 output_degree = [1], 
                 output_coef_init = [None], output_coef_train = [False], output_coef_reg = [[0.001, 1]], 
                 output_zero_order = [False], 
                 output_batch_norm = [False], 
                 output_regularize_linear = False, output_regularize_activation = False,
                 device = 'cpu', X_dtype = torch.float32, y_dtype = torch.float32):
            
        super(CustomRNN, self).__init__()
        
        # Store the arguments as attributes
        locals_ = locals().copy()
        for arg in locals_:
            if arg != 'self':
                setattr(self, arg, locals_[arg])
        
        if self.rnn_type == 'lstm':
          self.rnn = torch.nn.LSTM(input_size = self.input_size, 
                                   hidden_size = self.hidden_size, 
                                   num_layers = self.rnn_num_layers, 
                                   bias = self.rnn_bias, 
                                   batch_first = self.rnn_batch_first, 
                                   dropout = self.rnn_dropout, 
                                   bidirectional = self.rnn_bidirectional, 
                                   proj_size = self.lstm_proj_size, 
                                   device = self.device, dtype = self.X_dtype)
        if self.rnn_type == 'gru':
          self.rnn = torch.nn.GRU(input_size = self.input_size, 
                                   hidden_size = self.hidden_size, 
                                   num_layers = self.rnn_num_layers, 
                                   bias = self.rnn_bias, 
                                   batch_first = self.rnn_batch_first, 
                                   dropout = self.rnn_dropout, 
                                   bidirectional = self.rnn_bidirectional,       
                                   device = self.device, dtype = self.X_dtype)

        self.output_block = CustomSequential(in_features = self.hidden_size, 
                                             layer_out_features  = self.output_out_features, 
                                             layer_bias  = self.output_bias, 
                                             layer_weight_reg  = self.output_weight_reg, 
                                             layer_activation  = self.output_activation, 
                                             layer_degree  = self.output_degree, 
                                             layer_coef_init  = self.output_coef_init, 
                                             layer_coef_train  = self.output_coef_train, 
                                             layer_coef_reg  = self.output_coef_reg, 
                                             layer_zero_order  = self.output_zero_order, 
                                             layer_batch_norm  = self.output_batch_norm, 
                                             regularize_linear  = self.output_regularize_linear, 
                                             regularize_activation  = self.output_regularize_activation, 
                                             device  = self.device, dtype  = self.X_dtype)

    def forward(self, input, hiddens = None):
        
      input = input.clone().to(self.device, self.X_dtype)
      
      if not self.stateful:
        hiddens = None

      rnn_out, hiddens = self.rnn(input, hiddens)

      out = rnn_out[:, -self.output_len:, :]

      output = self.output_block(out).to(dtype = self.y_dtype)

      return output, hiddens
      
    def predict(self, input, hiddens = None):
        
        with torch.no_grad():
            output, hiddens = self.forward(input, hiddens = hiddens)
        
        return output, hiddens
                                  
    def penalty_score(self):
      penalty_fn = lambda reg, param: reg[0]*torch.norm(param.detach(), p = reg[1]) * int(param.requires_grad)
      
      penalty = 0

      if self.regularize_rnn:       
        for weight in [self.rnn.weight_ih_l0, self.rnn.weight_hh_l0]:
          penalty += penalty_fn(self.rnn_weight_reg, weight)

      if self.regularize_linear:
        penalty += penalty_fn(self.linear_weight_reg ,self.linear.weight)

      if self.output_activation == 'polynomial':
        penalty += self.output_activation_fn.regularize()
      
      return penalty

## Custom module employing multiple layers of linear transformation with nonlinear activation
class CustomSequential(torch.nn.Module):
  def __init__(self,
               in_features, layer_out_features = [1],
               layer_bias = [True],
               layer_weight_reg = [0.001, 1],
               layer_activation = ['relu'],
               layer_degree = [1],
               layer_coef_init = [None],
               layer_coef_train = [False],
               layer_coef_reg = [[0.001, 1]],
               layer_zero_order = [False],
               layer_batch_norm = [False],
               regularize_linear = False,
               regularize_activation = False,
               device = 'cpu', dtype = torch.float32):
  
    super(CustomSequential, self).__init__()
    
    # Store the arguments as attributes
    locals_ = locals().copy()
    for arg in locals_:
      if arg != 'self':
        if 'layer_' in arg:
          if len(locals_[arg]) == 1:
            locals_[arg] = locals_[arg] * len(layer_out_features)
          
        setattr(self, arg, locals_[arg])
    
    self.num_layers = len(self.layer_out_features)

    self.sequential = torch.nn.Sequential()
    for i in range(self.num_layers):
      if i == 0:
        in_features = self.in_features
      else:
        in_features = self.layer_out_features[i-1]

      self.sequential.add_module('linear_' + str(i+1),
                                 torch.nn.Linear(in_features = in_features, 
                                                 out_features = self.layer_out_features[i],
                                                 bias = self.layer_bias[i],
                                                 device = self.device, dtype = self.dtype))
      if self.layer_batch_norm[i]:
        self.sequential.add_module('batch_norm_' + str(i+1),
                                  torch.nn.BatchNorm1d(num_features = self.layer_out_features[i],
                                                       device = self.device, dtype = self.dtype))

      if self.layer_activation[i] == 'relu':
        self.sequential.add_module('activation_' + str(i+1),
                                   torch.nn.ReLU())
      elif self.layer_activation[i] == 'softmax':
        self.sequential.add_module('activation_' + str(i+1),
                                   torch.nn.Softmax(dim = -1))
      elif self.layer_activation[i] == 'sigmoid':
        self.sequential.add_module('activation_' + str(i+1),
                                   torch.nn.Sigmoid())
      elif self.layer_activation[i] == 'polynomial':
        self.sequential.add_module('activation_' + str(i+1),
                                   Polynomial(in_features = self.layer_out_features[i], 
                                              degree = self.layer_degree[i], 
                                              coef_init = self.layer_coef_init[i], 
                                              coef_train = self.layer_coef_train[i],
                                              coef_reg = self.layer_coef_reg[i], 
                                              zero_order = self.layer_zero_order[i], 
                                              device = self.device, dtype = self.dtype))
      
    else:
      self.activation_fn = torch.nn.Identity()

  def forward(self, input):
    output = self.sequential(input)
    return output

  def penalty_score(self):
    penalty_fn = lambda reg, param: reg[0]*torch.norm(param.detach(), p = reg[1]) * int(param.requires_grad)
    
    penalty = 0.
    if self.regularize_linear:
      for name, param in self.sequential.named_parameters():
        if ('linear' in name) & ('weight' in name):
          penalty += penalty_fn(self.layer_weight_reg, param)

    if self.regularize_activation:      
      for layer in self.sequential:
        if layer.__class__.__name__ == 'Polynomial':
          penalty += layer.penalty_score()

    return penalty

## Classification network using a simple linear block with nonlinear activation
class LinearActivationClassifier(torch.nn.Module):
    def __init__(self,
                 in_features, out_features = [1], activations = [None],
                 device = 'cpu', 
                 X_dtype = torch.float32, y_dtype = torch.float32):

        super(LinearActivationClassifier, self).__init__()

        self.device = device
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        
        self.in_features = in_features
        self.out_features = out_features
        self.activations = activations
        
        self.sequential = torch.nn.Sequential()

        for i in range(len(out_features)):
            
            if i == 0:
                in_features_i = in_features
            else:
                in_features_i = out_features[i-1]

            linear = torch.nn.Linear(in_features = in_features_i,
                                     out_features = out_features[i])
            
            if self.activations[i] == 'softmax':
                activation_fn = torch.nn.Softmax(dim = 1)
            elif self.activations[i] == 'sigmoid':
                activation_fn = torch.nn.Sigmoid()
            else:
                activation_fn = torch.nn.Identity()
            
            self.sequential.add_module(f"linear_{i}", linear)
            self.sequential.add_module(f"activation_{i}", activation_fn)
            
    def forward(self, input):

        input = input.clone().to(self.device, self.X_dtype)

        output = self.sequential(input).to(self.y_dtype)

        return output
    
    def predict_proba(self, input):

        with torch.no_grad():
            output = self.sequential(input).to(self.y_dtype)
        
        return output

    def predict(self, input, threshold = 0.5):

        output = self.predict_proba(input).to(int)
        
        return output

class Polynomial(torch.nn.Module):

  def __init__(self,
               in_features, degree = 1, coef_init = None, coef_train = True,
               coef_reg = [0.001, 1], zero_order = True,
               device = 'cpu', dtype = torch.float32):
      super(Polynomial, self).__init__()

      locals_ = locals().copy()

      for arg in locals_:
        if arg != 'self':
          setattr(self, arg, locals_[arg])
        
      # self.to(device = self.device, dtype = self.dtype)
      
      if self.coef_init is None:
          self.coef_init = torch.nn.init.normal_(torch.empty(self.in_features, self.degree + int(self.zero_order)))

      self.coef = torch.nn.Parameter(data = self.coef_init.to(device = self.device, dtype = self.dtype), requires_grad = self.coef_train)

  def forward(self, X):
 
    X = X.to(device = self.device, dtype = self.dtype)
    
    pows = torch.arange(1 - int(self.zero_order), (self.degree + 1), device = self.device, dtype = self.dtype)

    y = (X.unsqueeze(-1).pow(pows) * self.coef).sum(-1)

    return y

  def penalty_score(self):
    return self.coef_reg[0] * torch.norm(self.coef, p = self.coef_reg[1]) * int(self.coef.requires_grad)

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

        target = np.concatenate(self.target, 0)
        prediction = np.concatenate(self.prediction, 0)

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

        target = np.concatenate(self.target, 0)
        prediction = np.concatenate(self.prediction, 0)

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
