import torch
import numpy as np

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
