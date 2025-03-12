import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
    
    def forward(self, x, U, V):
        return torch.multiply(x, U) + torch.multiply(1-x, V)

class ImprovedMLP(nn.Module):
    def __init__(self, layers, activation_function, hard_constraint_fn=None, p_dropout=0.2, encoding=None):
        super(ImprovedMLP, self).__init__()
        
        self.layers = layers
        self.activation = activation_function
        self.hard_constraint_fn = hard_constraint_fn

        self.encoding = encoding
        if encoding:
            encoding.setup(self)

        '''U = φ(XW_1 + b_1)'''
        self.U = torch.nn.Sequential(nn.Linear(self.layers[0], self.layers[1]), self.activation())
        '''V = φ(XW_2 + b_2)'''
        self.V = torch.nn.Sequential(nn.Linear(self.layers[0], self.layers[1]), self.activation())

        # Livelli nascosti con connessioni residue
        self.hidden_layers = nn.ModuleList()
        for i in range(0, len(layers) - 2):
            '''Z_k = φ(H_k W_{z,k} + b_{z,k}) ∀ k = 1,...,L'''
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
            self.hidden_layers.append(self.activation())
            self.hidden_layers.append(Transformer())
            self.hidden_layers.append(nn.Dropout(p=p_dropout))

        '''f_θ(x) = H^(L+1)W + b'''
        self.output_layer = nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        orig_x = x

        if self.encoding:
            x = self.encoding(x)

        # Genera trasformazioni U e V
        U = self.U(orig_x)
        V = self.V(orig_x)

        # Propagazione nei livelli nascosti con connessioni residue
        output = x
        for i in range(0, len(self.hidden_layers), 4):  # 4: Linear -> Activation -> Transformer -> Dropout
            '''Z_k = φ(H_k W_{z,k} + b_{z,k})    ∀ k = 1,...,L'''
            Z = self.hidden_layers[i](output) #Linear
            output = self.hidden_layers[i + 1](Z)  # Activation
            '''H_{k+1} = (1 - Z_k) ⊙ U + Z_k ⊙ V    ∀ k = 1,...,L'''
            H = self.hidden_layers[i + 2](Z, U, V) #Transformer
            output = self.hidden_layers[i + 3](H)  # Dropout


        # Propagazione finale
        '''f_θ(x) = H^(L+1)W + b'''
        F = self.output_layer(H)

        if self.hard_constraint_fn:
            output = self.hard_constraint_fn(orig_x, F)

        return output					

class ImprovedEncodedMlp(nn.Module):
    def __init__(self, layers, activation_function, hard_constraint_fn=None, p_dropout=0.2, encoding=None):
        super(ImprovedEncodedMlp, self).__init__()
        
        self.layers = layers
        self.activation = activation_function
        self.hard_constraint_fn = hard_constraint_fn

        self.encoding = encoding
        if encoding:
            encoding.setup(self)

        '''U = φ(XW_1 + b_1)'''
        self.U = torch.nn.Sequential(nn.Linear(self.layers[0], self.layers[1]), self.activation())
        '''V = φ(XW_2 + b_2)'''
        self.V = torch.nn.Sequential(nn.Linear(self.layers[0], self.layers[1]), self.activation())

        # Livelli nascosti con connessioni residue
        self.hidden_layers = nn.ModuleList()
        for i in range(0, len(layers) - 2):
            '''Z_k = φ(H_k W_{z,k} + b_{z,k}) ∀ k = 1,...,L'''
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
            self.hidden_layers.append(self.activation())
            self.hidden_layers.append(Transformer())
            self.hidden_layers.append(nn.Dropout(p=p_dropout))

        '''f_θ(x) = H^(L+1)W + b'''
        self.output_layer = nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        orig_x = x

        if self.encoding:
            x = self.encoding(x)

        # Genera trasformazioni U e V
        U = self.U(x)
        V = self.V(x)

        # Propagazione nei livelli nascosti con connessioni residue
        output = x
        for i in range(0, len(self.hidden_layers), 4):  # 4: Linear -> Activation -> Transformer -> Dropout
            '''Z_k = φ(H_k W_{z,k} + b_{z,k})    ∀ k = 1,...,L'''
            Z = self.hidden_layers[i](output) #Linear
            output = self.hidden_layers[i + 1](Z)  # Activation
            '''H_{k+1} = (1 - Z_k) ⊙ U + Z_k ⊙ V    ∀ k = 1,...,L'''
            H = self.hidden_layers[i + 2](Z, U, V) #Transformer
            output = self.hidden_layers[i + 3](H)  # Dropout


        # Propagazione finale
        '''f_θ(x) = H^(L+1)W + b'''
        F = self.output_layer(H)

        if self.hard_constraint_fn:
            output = self.hard_constraint_fn(orig_x, F)

        return output	



class FourierFeatureEncoding(nn.Module):
    """Fourier feature mapping for coordinate encoding."""

    def __init__(self, sigma: float, input_size: int, encoded_size: int):
        """
        Args:
            sigma (float): Scaling factor for Fourier features.
            input_size (int): Number of input dimensions.
            encoded_size (int): Number of Fourier features per input dimension.
        """
        super(FourierFeatureEncoding, self).__init__()
        self.sigma = sigma
        self.input_size = input_size
        self.encoded_size = encoded_size
        self.B = None  # Placeholder, will be initialized in `setup`

    def setup(self, model):
        """Sets up the encoding and modifies the MLP input layer size."""
        if self.B is None:
            self.B = torch.randn((self.encoded_size, self.input_size)) * self.sigma
        self.register_buffer('B_buffer', self.B)  # Ensure it's not trainable
        model.layers[0] = self.encoded_size * 2  # Update MLP input layer size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input coordinates using Fourier features."""
        x_proj = 2 * torch.pi * x @ self.B_buffer.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    


