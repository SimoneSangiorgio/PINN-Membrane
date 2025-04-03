import torch
import torch.nn as nn
import numpy as np
from pinns_v2.rff import GaussianEncoding
from collections import OrderedDict
import math


class ModifiedMLP(nn.Module):
    def __init__(self, layers, activation_function, hard_constraint_fn=None, p_dropout=0.2, encoding=None) -> None:
        super(ModifiedMLP, self).__init__()

        self.layers = layers
        self.activation = activation_function
        self.encoding = encoding
        if encoding != None:
            encoding.setup(self)
        
        self.U = torch.nn.Sequential(nn.Linear(self.layers[0], self.layers[1]), self.activation())
        self.V = torch.nn.Sequential(nn.Linear(self.layers[0], self.layers[1]), self.activation())

        layer_list = nn.ModuleList()        
        for i in range(0, len(self.layers)-2):
            layer_list.append(
                nn.Linear(layers[i], layers[i+1])
            )
            layer_list.append(self.activation())
            layer_list.append(Transformer())
            layer_list.append(nn.Dropout(p = p_dropout))
        self.hidden_layer = layer_list
        self.output_layer = nn.Linear(self.layers[-2], self.layers[-1])

        self.hard_constraint_fn = hard_constraint_fn
        

    def forward(self, x):
        orig_x = x
        if self.encoding != None:
            x = self.encoding(x)

        U = self.U(orig_x)
        V = self.V(orig_x)

        output = x
        for i in range(0, len(self.hidden_layer), 4):
            output = self.hidden_layer[i](output) #Linear
            output = self.hidden_layer[i+1](output) #Activation
            output = self.hidden_layer[i+2](output, U, V) #Transformer
            output = self.hidden_layer[i+3](output) #Dropout
        output = self.output_layer(output)

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
    

class TimeFourierMLP(nn.Module):
    def __init__(self, layers, activation_function, sigma, encoded_size, hard_constraint_fn=None, p_dropout=0.2) -> None:
        super(TimeFourierMLP, self).__init__()

        orig_initial_layer = layers[0]
        self.layers = layers
        self.activation = activation_function
        self.encoding = GaussianEncoding(sigma = sigma, input_size=1, encoded_size=encoded_size)
        self.encoding.setup(self)
        # restore layer size for the first layer of the MLP
        # the first layer size should be the encoding of t (with encoded_size*2 size)
        # and the other components of the input not encoded
        self.layers[0] = encoded_size*2 + orig_initial_layer - 1 

        layer_list = list()        
        for i in range(len(self.layers)-2):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(self.layers[i], self.layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('dropout_%d' % i, nn.Dropout(p = p_dropout)))
        layer_list.append(('layer_%d' % (len(self.layers)-1), nn.Linear(self.layers[-2], self.layers[-1])))

        self.mlp = nn.Sequential(OrderedDict(layer_list))

        self.hard_constraint_fn = hard_constraint_fn

    def forward(self, x):
        orig_x = x
        x = self.encoding(x[-1:]) # just encode time component
        x = torch.cat((orig_x[:-1], x), dim=0) # concatenate with the other components of the input
        output = self.mlp(x)

        if self.hard_constraint_fn != None:
            output = self.hard_constraint_fn(orig_x, output)

        return output




class NormalizedMLP(nn.Module):
    def __init__(self, layers, activation_function, range_input, hard_constraint_fn=None, p_dropout=0.2, encoding=None) -> None:
        super(NormalizedMLP, self).__init__()

        self.norm = Normalization_strat(range_input.clone().detach()) 
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
        x = self.norm(x)
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

class Normalization_strat(nn.Module):
    def __init__(self, tensor_range):
        super(Normalization_strat, self).__init__()
        self.tensor_range = tensor_range

    def forward(self, x):
        return (x/(self.tensor_range+1e-5))    # element-wise division at each index by tensor_range, for every element of x (a small number 0.00001 is added to handle 0 denominator cases)
					