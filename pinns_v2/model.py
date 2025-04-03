import torch
import torch.nn as nn
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
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
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
            #self.hidden_layers.append(Transformer())
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
        for i in range(0, len(self.hidden_layers), 4):  # 4: Linear -> Activation -> Transformer -> Dropout
            '''Z_k = φ(H_k W_{z,k} + b_{z,k})    ∀ k = 1,...,L'''
            Z = self.hidden_layers[i](x) #Linear
            Z = self.hidden_layers[i + 1](Z)  # Activation
            #Z = self.hidden_layers[i + 2](Z, U, V) #Transformer
            Z = self.hidden_layers[i + 2](Z)  # Dropout

            '''H_{k+1} = (1 - Z_k) ⊙ U + Z_k ⊙ V    ∀ k = 1,...,L'''
            x = (1 - Z) * U + Z * V

        # Propagazione finale
        '''f_θ(x) = H^(L+1)W + b'''
        x = self.output_layer(x)

        if self.hard_constraint_fn:
            x = self.hard_constraint_fn(orig_x, x)

        return x					
    
# class SimpleSpatioTemporalFFN(nn.Module):
#     def __init__(self, spatial_sigmas, temporal_sigmas, hidden_layers, activation, hard_constraint_fn=None):
#         """
#         Simplified version of paper's architecture (section 3.3)
#         - Input format: [..., 3] where last dim is (x, y, t)
#         - Uses registered buffers for B matrices for correct device handling.
#         """
#         super().__init__()
#         self.spatial_dim = 2
#         self.temporal_dim = 1
#         self.spatial_sigmas = spatial_sigmas # Store sigmas for reference if needed
#         self.temporal_sigmas = temporal_sigmas
#         self.hard_constraint_fn = hard_constraint_fn

#         if not hidden_layers:
#              raise ValueError("hidden_layers list cannot be empty")

#         # --- Register B matrices as buffers ---
#         # Use unique names for each buffer
#         for i, s in enumerate(spatial_sigmas):
#             # Create tensor (on CPU initially is fine)
#             b_matrix = torch.randn(hidden_layers[0] // 2, self.spatial_dim) * s
#             # Register it as a buffer
#             self.register_buffer(f"spatial_B_{i}", b_matrix)

#         for i, s in enumerate(temporal_sigmas):
#             b_matrix = torch.randn(hidden_layers[0] // 2, self.temporal_dim) * s
#             self.register_buffer(f"temporal_B_{i}", b_matrix)
#         # --- End Buffer Registration ---

#         # Shared MLP
#         self.mlp = nn.Sequential()
#         # Input size to MLP is double the number of frequencies (cos + sin)
#         mlp_input_size = hidden_layers[0]
#         current_size = mlp_input_size
#         for i in range(len(hidden_layers) - 1):
#             self.mlp.append(nn.Linear(current_size, hidden_layers[i+1]))
#             self.mlp.append(activation())
#             current_size = hidden_layers[i+1]

#         # Final layer input size calculation
#         final_mlp_output_size = hidden_layers[-1]
#         final_layer_input_size = len(spatial_sigmas) * len(temporal_sigmas) * final_mlp_output_size
#         self.final = nn.Linear(final_layer_input_size, 1)

#     def forward(self, x):
#         # Split input
#         x_spatial = x[..., :self.spatial_dim]
#         x_time = x[..., self.spatial_dim:] # Use slicing relative to spatial_dim

#         # Process spatial features
#         spatial_feats = []
#         # Iterate through the registered buffers
#         for i in range(len(self.spatial_sigmas)):
#             # Retrieve the buffer by its registered name
#             B = getattr(self, f"spatial_B_{i}")
#             # Ensure B is on the same device as x_spatial (buffers are moved by .to(device))
#             proj = 2 * math.pi * x_spatial @ B.T
#             encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
#             spatial_feats.append(self.mlp(encoded))

#         # Process temporal features
#         temporal_feats = []
#         for i in range(len(self.temporal_sigmas)):
#             B = getattr(self, f"temporal_B_{i}")
#             proj = 2 * math.pi * x_time @ B.T
#             encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
#             temporal_feats.append(self.mlp(encoded))

#         # Combine features: element-wise product for each pair
#         combined_features = []
#         for s_feat in spatial_feats:
#             for t_feat in temporal_feats:
#                 combined_features.append(s_feat * t_feat) # Element-wise multiplication

#         # Concatenate results and pass through final layer
#         out = self.final(torch.cat(combined_features, dim=-1))

#         if self.hard_constraint_fn:
#             out = self.hard_constraint_fn(x, out) # Pass original combined input

#         return out
class SimpleSpatioTemporalFFN(nn.Module):
    def __init__(self, spatial_sigmas, temporal_sigmas, hidden_layers, activation, hard_constraint_fn=None):
        """
        Simplified version of paper's architecture (section 3.3)
        - Input format: [..., 3] where last dim is (x, y, t)
        - No special wrapper handling
        - Clean separation of spatial/temporal processing
        """
        super().__init__()
        self.spatial_dim = 2
        self.temporal_dim = 1
        
        # Fourier feature matrices
        self.spatial_B = [torch.randn(hidden_layers[0]//2, 2) * s for s in spatial_sigmas]
        self.temporal_B = [torch.randn(hidden_layers[0]//2, 1) * s for s in temporal_sigmas]
        self.hard_constraint_fn = hard_constraint_fn
        # Shared MLP
        self.mlp = nn.Sequential()
        for i in range(len(hidden_layers)-1):
            self.mlp.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.mlp.append(activation())
            
        # Final layer
        self.final = nn.Linear(
            len(spatial_sigmas)*len(temporal_sigmas)*hidden_layers[-1], 
            1
        )

    def forward(self, x):
        # Split input (last dimension must be 3)
        x_spatial = x[..., :2]  # x, y
        x_time = x[..., 2:]     # t
        
        # Process spatial features
        spatial_feats = []
        for B in self.spatial_B:
            proj = 2*math.pi * x_spatial @ B.T
            encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            spatial_feats.append(self.mlp(encoded))
            
        # Process temporal features
        temporal_feats = []
        for B in self.temporal_B:
            proj = 2*math.pi * x_time @ B.T
            encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            temporal_feats.append(self.mlp(encoded))
            
        # Combine features
        combined = [s*t for s in spatial_feats for t in temporal_feats]
        out = self.final(torch.cat(combined, dim=-1))
        if self.hard_constraint_fn:
            out = self.hard_constraint_fn(x, out)
       
        return out