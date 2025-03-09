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
    


class FourierFeatureEncoding(nn.Module):
    def __init__(self, sigma: Optional[float] = None,
                 input_size: Optional[int] = None,
                 encoded_size: Optional[int] = None,
                 b: Optional[torch.Tensor] = None):
        super(FourierFeatureEncoding, self).__init__()
        self.b = b
        self.sigma = sigma
        self.encoded_size = encoded_size
        self.input_size = input_size

        if self.b is None:
            if self.sigma is None or self.input_size is None or self.encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.')
        elif self.sigma is not None or self.input_size is not None or self.encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')

    def setup(self, model):
        if self.b is None:
            self.b = torch.randn(self.encoded_size, self.input_size) * self.sigma
        self.register_buffer('enc', self.b)
        model.layers[0] = self.encoded_size * 2

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * torch.pi * v @ self.b.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# Esempio di utilizzo
if __name__ == "__main__":
    batch_size = 10
    input_size = 3  # Esempio per x, y
    encoded_size = 64  # Numero di feature di Fourier
    sigma = 10.0  # Definisce la scala delle feature
    
    encoder = FourierFeatureEncoding(input_size, encoded_size, sigma)
    x = torch.rand(batch_size, input_size)  # Input random
    encoded_x = encoder(x)
    print("Input shape:", x.shape)
    print("Encoded shape:", encoded_x.shape)

