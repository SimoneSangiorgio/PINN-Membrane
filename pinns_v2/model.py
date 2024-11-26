import torch
import torch.nn as nn
import numpy as np
from pinns_v2.rff import GaussianEncoding
from collections import OrderedDict
import math


import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

class ModifiedMLP(nn.Module):
    def __init__(self, layers, activation_function, hard_constraint_fn=None, p_dropout=0.2, encoding=None) -> None:
        super(ModifiedMLP, self).__init__()

        self.layers = layers
        self.activation = activation_function
        self.encoding = encoding
        if encoding != None:
            encoding.setup(self)
        
        # Transformer networks U and V (equation 43)
        self.U = nn.Sequential(
            nn.Linear(self.layers[0], self.layers[1]), 
            self.activation()
        )
        self.V = nn.Sequential(
            nn.Linear(self.layers[0], self.layers[1]), 
            self.activation()
        )

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        
        # First hidden layer - H⁽¹⁾ = φ(XWz,1 + bz,1)
        self.first_layer = nn.Sequential(
            nn.Linear(self.layers[0], self.layers[1]),
            self.activation()
        )
        
        # Z layers - Z⁽ᵏ⁾ = φ(H⁽ᵏ⁾Wz,k + bz,k)
        for i in range(len(self.layers)-2):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(self.layers[1], self.layers[1]),
                    self.activation(),
                    nn.Dropout(p=p_dropout)
                )
            )
            
        # Output layer
        self.output_layer = nn.Linear(self.layers[1], self.layers[-1])
        
        self.hard_constraint_fn = hard_constraint_fn

    def forward(self, x):
        orig_x = x
        if self.encoding != None:
            x = self.encoding(x)

        # Calculate U and V (equation 43)
        U = self.U(orig_x)
        V = self.V(orig_x)
        
        # First hidden layer (equation 44)
        H = self.first_layer(x)
        
        # Process through hidden layers (equations 45-46)
        for layer in self.hidden_layers:
            # Calculate Z⁽ᵏ⁾ = φ(H⁽ᵏ⁾Wz,k + bz,k)
            Z = layer(H)
            # Calculate H⁽ᵏ⁺¹⁾ = (1-Z⁽ᵏ⁾)U + Z⁽ᵏ⁾V
            H = torch.multiply(1-Z, U) + torch.multiply(Z, V)
        
        # Output layer (equation 47)
        output = self.output_layer(H)

        if self.hard_constraint_fn != None:
            output = self.hard_constraint_fn(orig_x, output)

        return output

class MLP(nn.Module):
    def __init__(self, layers, activation_function, hard_constraint_fn=None, p_dropout=0.2, encoding=None) -> None:
        super(MLP, self).__init__()

        self.layers = layers
        self.activation = activation_function
        self.encoding = encoding
        if encoding != None:
            encoding.setup(self)

        layer_list = list()        
        for i in range(len(self.layers)-2):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('dropout_%d' % i, nn.Dropout(p = p_dropout)))
        layer_list.append(('layer_%d' % (len(self.layers)-1), nn.Linear(self.layers[-2], self.layers[-1])))

        self.mlp = nn.Sequential(OrderedDict(layer_list))

        self.hard_constraint_fn = hard_constraint_fn

    def forward(self, x):
        orig_x = x
        if self.encoding != None:
            x = self.encoding(x)

        output = self.mlp(x)

        if self.hard_constraint_fn != None:
            output = self.hard_constraint_fn(orig_x, output)

        return output


class Sin(nn.Module):
  def __init__(self):
    super(Sin, self).__init__()

  def forward(self, x):
    return torch.sin(x)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
    
    def forward(self, x, U, V):
        return torch.multiply(x, U) + torch.multiply(1-x, V)
        #return torch.nn.functional.linear(torch.multiply(x, U) + torch.multiply((1-x), V), self.weight, self.bias)

class FactorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, sigma = 0.1, mu = 1.0):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = nn.init.xavier_normal_(self.weight)
        self.s = (torch.randn(self.in_features) * sigma) + mu
        self.s = torch.exp(self.s)
        self.v = self.weight/self.s
        self.s = nn.parameter.Parameter(self.s)
        self.v = nn.parameter.Parameter(self.v)
        if bias:
            self.bias = nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.s*self.v, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class FactorizedModifiedLinear(FactorizedLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
    
    def forward(self, x, U , V):
        return torch.nn.functional.linear(torch.multiply(x, U) + torch.multiply((1-x), V), self.s*self.v, self.bias)


					