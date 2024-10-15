import torch
import torch.nn as nn
import numpy as np
from pinns_v2.rff import GaussianEncoding
from collections import OrderedDict
import math

class PINN(nn.Module):

    def __init__(self, layers, activation_function, hard_constraint_fn = None, modified_MLP = False, ff=False, sigma = None):
        super(PINN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = activation_function 
        self.ff = ff      
        layer_list = list()
        if self.ff:
            if sigma == None:
                return ValueError("If Random Fourier Features embedding is on, then a sigma must be specified")
            self.encoding = GaussianEncoding(sigma=sigma, input_size=layers[0], encoded_size=layers[0])
            layers[0] *= 2

        if modified_MLP:
            self.U = torch.nn.ModuleList([torch.nn.Linear(layers[0], layers[1]), self.activation()])
            self.V = torch.nn.ModuleList([torch.nn.Linear(layers[0], layers[1]), self.activation()])
            layer_list.append(torch.nn.Linear(layers[0], layers[1]))
            layer_list.append(self.activation())
            for i in range(1, self.depth - 1):
                layer_list.append(ModifiedMLP(layers[i], layers[i+1]))
                layer_list.append(self.activation())
            layer_list.append(torch.nn.Linear(layers[-2], layers[-1]))
        else:
            for i in range(self.depth - 1):
                layer_list.append(
                    ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
                )
                layer_list.append(('activation_%d' % i, self.activation()))
                
            layer_list.append(
                ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
            )
            layerDict = OrderedDict(layer_list)
        
        # deploy layers
        if modified_MLP:
            self.layers = torch.nn.ModuleList(layer_list)
        else:
            self.layers = torch.nn.Sequential(layerDict)
        self.modified_MLP = modified_MLP
        self.hard_constraint_fn = hard_constraint_fn
		
    def forward(self, x):
        try:
            if self.ff:
                x = self.encoding(x)
        except:
            pass

        if self.modified_MLP:
            U = self.U[1](self.U[0](x))
            V = self.V[1](self.V[0](x))
            output = self.layers[1](self.layers[0](x))
            for i in range(2, len(self.layers) - 2, 2):
                output = self.layers[i+1](self.layers[i](output, U, V))
            output = self.layers[-1](output)
        else:
            output = self.layers(x)
        if self.hard_constraint_fn != None:
            output = self.hard_constraint_fn(x, output)
        #d = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), only_inputs=True, create_graph=True)[0]
        #output = torch.hstack((output, d))
        return output


class Sin(nn.Module):
  def __init__(self):
    super(Sin, self).__init__()

  def forward(self, x):
    return torch.sin(x)
  

class ModifiedMLP(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
    
    def forward(self, x, U, V):
        return torch.nn.functional.linear(torch.multiply(x, U) + torch.multiply((1-x), V), self.weight, self.bias)
	
					