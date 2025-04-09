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
					

class SimpleSpatioTemporalFFN(nn.Module):
    def __init__(self, spatial_sigmas, temporal_sigmas, hidden_layers, activation, hard_constraint_fn=None):
        """
        Simplified version of paper's architecture (section 3.3)
        - Input format: [..., 3] where last dim is (x, y, t)
        - Uses registered buffers for B matrices for correct device handling.
        """
        super().__init__()
        self.spatial_dim = 2
        self.temporal_dim = 1
        self.spatial_sigmas = spatial_sigmas # Store sigmas for reference if needed
        self.temporal_sigmas = temporal_sigmas
        self.hard_constraint_fn = hard_constraint_fn

        if not hidden_layers:
             raise ValueError("hidden_layers list cannot be empty")

        # --- Register B matrices as buffers ---
        # Use unique names for each buffer
        for i, s in enumerate(spatial_sigmas):
            # Create tensor (on CPU initially is fine)
            b_matrix = torch.randn(hidden_layers[0] // 2, self.spatial_dim) * s
            # Register it as a buffer
            self.register_buffer(f"spatial_B_{i}", b_matrix)

        for i, s in enumerate(temporal_sigmas):
            b_matrix = torch.randn(hidden_layers[0] // 2, self.temporal_dim) * s
            self.register_buffer(f"temporal_B_{i}", b_matrix)
        # --- End Buffer Registration ---

        # Shared MLP
        self.mlp = nn.Sequential()
        # Input size to MLP is double the number of frequencies (cos + sin)
        mlp_input_size = hidden_layers[0]
        current_size = mlp_input_size
        for i in range(len(hidden_layers) - 1):
            self.mlp.append(nn.Linear(current_size, hidden_layers[i+1]))
            self.mlp.append(activation())
            current_size = hidden_layers[i+1]

        # Final layer input size calculation
        final_mlp_output_size = hidden_layers[-1]
        final_layer_input_size = len(spatial_sigmas) * len(temporal_sigmas) * final_mlp_output_size
        self.final = nn.Linear(final_layer_input_size, 1)

    def forward(self, x):
        # Split input
        x_spatial = x[..., :self.spatial_dim]
        x_time = x[..., self.spatial_dim:] # Use slicing relative to spatial_dim

        # Process spatial features
        spatial_feats = []
        # Iterate through the registered buffers
        for i in range(len(self.spatial_sigmas)):
            # Retrieve the buffer by its registered name
            B = getattr(self, f"spatial_B_{i}")
            # Ensure B is on the same device as x_spatial (buffers are moved by .to(device))
            proj = 2 * math.pi * x_spatial @ B.T
            encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            spatial_feats.append(self.mlp(encoded))

        # Process temporal features
        temporal_feats = []
        for i in range(len(self.temporal_sigmas)):
            B = getattr(self, f"temporal_B_{i}")
            proj = 2 * math.pi * x_time @ B.T
            encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            temporal_feats.append(self.mlp(encoded))

        # Combine features: element-wise product for each pair
        combined_features = []
        for s_feat in spatial_feats:
            for t_feat in temporal_feats:
                combined_features.append(s_feat * t_feat) # Element-wise multiplication

        # Concatenate results and pass through final layer
        out = self.final(torch.cat(combined_features, dim=-1))

        if self.hard_constraint_fn:
            out = self.hard_constraint_fn(x, out) # Pass original combined input

        return out


class SpatioTemporalFFN(nn.Module): 
    def __init__(self,
                 spatial_feature_indices: list[int],
                 temporal_indices: list[int],
                 spatial_sigmas: list[float],
                 temporal_sigmas: list[float],
                 # hidden_layers defines FFN output AND shared MLP structure
                 hidden_layers: list[int], # E.g., [200, 200, 200, 200]
                 activation: nn.Module,
                 hard_constraint_fn=None):
        """
        Spatio-Temporal Fourier Feature Network (based on Fig 8b, arXiv:2012.10047v1).
        Mimics the structure of SimpleSpatioTemporalFFN reference but uses
        indices to select features/coordinates for spatial and temporal paths.
        Encodes features specified by spatial_feature_indices spatially,
        and coordinates specified by temporal_indices temporally.

        Args:
            spatial_feature_indices (list[int]): Indices for features encoded spatially.
            temporal_indices (list[int]): Indices for temporal coordinates.
            spatial_sigmas (list[float]): Sigmas for spatial feature FFN.
            temporal_sigmas (list[float]): Sigmas for temporal coordinate FFN.
            hidden_layers (list[int]): Defines network structure.
                                       hidden_layers[0]: Output size of FFN encoding (must be even)
                                                         AND input size to the first shared MLP layer.
                                       hidden_layers[1:]: Sizes of subsequent layers in the shared MLP.
            activation (nn.Module): Activation function class for the MLP.
            hard_constraint_fn (callable, optional): Applied to original input and network output.
        """
        super().__init__() # Call superclass init

        if not hidden_layers or len(hidden_layers) < 2:
            raise ValueError("hidden_layers must have at least two elements (FFN size, MLP output size).")
        if hidden_layers[0] % 2 != 0:
             print(f"Warning: FFN output size ({hidden_layers[0]}) should ideally be even.")

        self.spatial_sigmas = spatial_sigmas if spatial_sigmas else []
        self.temporal_sigmas = temporal_sigmas if temporal_sigmas else []
        self.spatial_feature_indices = spatial_feature_indices
        self.temporal_indices = temporal_indices
        self.hard_constraint_fn = hard_constraint_fn

        self.num_spatial_features = len(self.spatial_feature_indices)
        self.num_temporal_dims = len(self.temporal_indices)
        self.ffn_output_size = hidden_layers[0] # e.g., 200
        self.num_freq = self.ffn_output_size // 2 # e.g., 100

        # --- Register B matrices as buffers ---
        if self.num_spatial_features > 0 and self.spatial_sigmas:
            for i, s in enumerate(self.spatial_sigmas):
                b_matrix = torch.randn(self.num_freq, self.num_spatial_features) * s
                self.register_buffer(f"spatial_B_{i}", b_matrix)
        if self.num_temporal_dims > 0 and self.temporal_sigmas:
             for i, s in enumerate(self.temporal_sigmas):
                b_matrix = torch.randn(self.num_freq, self.num_temporal_dims) * s
                self.register_buffer(f"temporal_B_{i}", b_matrix)
        # --- End B matrices ---

        # --- Shared MLP ---
        self.mlp = nn.Sequential()
        current_size = self.ffn_output_size
        if len(hidden_layers) > 1:
            for i in range(len(hidden_layers) - 1):
                self.mlp.append(nn.Linear(current_size, hidden_layers[i+1]))
                self.mlp.append(activation())
                current_size = hidden_layers[i+1]
        self.mlp_output_size = current_size
        # --- End MLP ---

        # --- Final Layer ---
        num_spatial_branches = max(1, len(self.spatial_sigmas)) if self.num_spatial_features > 0 else 1
        num_temporal_branches = max(1, len(self.temporal_sigmas)) if self.num_temporal_dims > 0 else 1
        final_layer_input_size = num_spatial_branches * num_temporal_branches * self.mlp_output_size
        self.final = nn.Linear(final_layer_input_size, 1)
        # --- End Final Layer ---

    def forward(self, x):
        """Forward pass mimicking SimpleSpatioTemporalFFN structure."""
        original_input = x # Keep for hard constraints

        # --- Select feature subsets based on indices ---
        x_spatial_features = x[..., self.spatial_feature_indices] if self.num_spatial_features > 0 else None
        x_temporal_coords = x[..., self.temporal_indices] if self.num_temporal_dims > 0 else None

        # --- Process spatial features ---
        spatial_mlp_outputs = []
        if x_spatial_features is not None and self.spatial_sigmas:
            for i in range(len(self.spatial_sigmas)):
                B = getattr(self, f"spatial_B_{i}")
                proj = 2 * math.pi * x_spatial_features @ B.T
                encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
                processed = self.mlp(encoded)
                spatial_mlp_outputs.append(processed)
        else:
            placeholder_shape = list(x.shape[:-1]) + [self.mlp_output_size]
            placeholder = torch.ones(placeholder_shape, device=x.device)
            spatial_mlp_outputs.append(placeholder)

        # --- Process temporal features ---
        temporal_mlp_outputs = []
        if x_temporal_coords is not None and self.temporal_sigmas:
            for i in range(len(self.temporal_sigmas)):
                B = getattr(self, f"temporal_B_{i}")
                proj = 2 * math.pi * x_temporal_coords @ B.T
                encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
                processed = self.mlp(encoded)
                temporal_mlp_outputs.append(processed)
        else:
            placeholder_shape = list(x.shape[:-1]) + [self.mlp_output_size]
            placeholder = torch.ones(placeholder_shape, device=x.device)
            temporal_mlp_outputs.append(placeholder)

        # --- Combine features: element-wise product for each pair ---
        combined_features = []
        for s_feat in spatial_mlp_outputs:
            for t_feat in temporal_mlp_outputs:
                combined_features.append(s_feat * t_feat)

        # --- Concatenate results and pass through final layer ---
        final_input_tensor = torch.cat(combined_features, dim=-1)
        out = self.final(final_input_tensor)

        # --- Apply hard constraints ---
        if self.hard_constraint_fn:
            out = self.hard_constraint_fn(original_input, out)

        return out