import torch
import torch.nn as nn
import math

class SpatioTemporalFFN(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, spatial_sigmas, temporal_sigmas, 
                 hidden_layers, activation, hard_constraint_fn=None):
        """
        Final corrected implementation handling:
        - Gradient tracking wrappers
        - BatchedTensor dimensions
        - 1D/2D input variations
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.total_dim = spatial_dim + temporal_dim
        self.hard_constraint_fn = hard_constraint_fn

        # Initialize Fourier feature matrices (Equation 3.29-3.30)
        self.M_x = len(spatial_sigmas)
        self.M_t = len(temporal_sigmas)
        
        # Spatial encodings B_x (paper uses fixed Gaussian matrices)
        self.spatial_B = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_layers[0]//2, spatial_dim) * sigma, 
                        requires_grad=False) 
            for sigma in spatial_sigmas
        ])
        
        # Temporal encodings B_t (paper uses fixed Gaussian matrices)
        self.temporal_B = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_layers[0]//2, temporal_dim) * sigma, 
                        requires_grad=False)
            for sigma in temporal_sigmas
        ])

        # Shared MLP (Equation 3.31-3.32)
        self.shared_mlp = nn.Sequential()
        for i in range(len(hidden_layers)-1):
            self.shared_mlp.add_module(f'layer_{i}', 
                                     nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.shared_mlp.add_module(f'act_{i}', activation())

        # Final linear layer (Equation 3.34)
        final_in_dim = self.M_x * self.M_t * hidden_layers[-1]
        self.final_layer = nn.Linear(final_in_dim, 1)

    def forward(self, x):
        """
        Handles:
        - GradTrackingTensor/BatchedTensor wrappers
        - 1D inputs (no batch dimension)
        - Automatic dimension validation
        """
        # Preserve original tensor type and wrappers
        orig_type = type(x)
        needs_squeeze = False
        
        # Handle 1D inputs (no batch dimension)
        if x.ndim == 1:
            x = x.unsqueeze(0)
            needs_squeeze = True
            
        # Validate dimensions through wrappers
        try:
            # Access underlying tensor if wrapped
            if hasattr(x, 'value'):
                last_dim = x.value.shape[-1]
            else:
                last_dim = x.shape[-1]
                
            assert last_dim == self.total_dim, \
                f"Input last dim {last_dim} != {self.total_dim} (spatial {self.spatial_dim} + temporal {self.temporal_dim})"
        except AttributeError:
            pass

        # Extract coordinates using ellipsis for wrapper compatibility
        spatial_coords = x[..., :self.spatial_dim]
        temporal_coords = x[..., self.spatial_dim:self.total_dim]

        # Process spatial features (Equation 3.29, 3.31)
        spatial_features = []
        for B_x in self.spatial_B:
            # Use torch.einsum for wrapper compatibility
            proj = 2 * math.pi * torch.einsum('...i,ji->...j', spatial_coords, B_x)
            encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            h = self.shared_mlp(encoded)
            spatial_features.append(h)

        # Process temporal features (Equation 3.30, 3.32)
        temporal_features = []
        for B_t in self.temporal_B:
            proj = 2 * math.pi * torch.einsum('...i,ji->...j', temporal_coords, B_t)
            encoded = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            h = self.shared_mlp(encoded)
            temporal_features.append(h)

        # Combine via element-wise multiplication (Equation 3.33)
        combined = []
        for h_x in spatial_features:
            for h_t in temporal_features:
                combined.append(h_x * h_t)  # Point-wise multiplication

        # Final output (Equation 3.34)
        out = self.final_layer(torch.cat(combined, dim=-1))

        # Apply hard constraint if provided
        if self.hard_constraint_fn:
            out = self.hard_constraint_fn(x, out)

        # Restore original shape and tensor type
        if needs_squeeze:
            out = out.squeeze(0)
            
        return orig_type(out) if orig_type != torch.Tensor else out
    

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