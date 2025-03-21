import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalFFN(nn.Module):
    """
    Spatio-temporal multi-scale Fourier feature network.
    
    Args:
        spatial_dim (int): Dimension of spatial input (e.g., 2 for x,y).
        num_spatial_features (int): Number of spatial Fourier embeddings (M_x).
        num_temporal_features (int): Number of temporal Fourier embeddings (M_t).
        m (int): Number of Fourier features per embedding (B matrix rows).
        sigma_x (float): Scale for spatial Fourier features initialization.
        sigma_t (float): Scale for temporal Fourier features initialization.
        hidden_layers (int): Number of hidden layers in the shared FC network.
        hidden_size (int): Size of each hidden layer.
        output_size (int): Size of the output (e.g., 1 for scalar PDE solutions).
    """
    def __init__(self, spatial_dim, num_spatial_features, num_temporal_features, 
                 m, sigma_x, sigma_t, hidden_layers, hidden_size, output_size):
        super().__init__()
        self.m = m
        self.Mx = num_spatial_features
        self.Mt = num_temporal_features
        
        # Initialize spatial Fourier feature matrices B_x (fixed during training)
        self.B_x = nn.ParameterList([
            nn.Parameter(sigma_x * torch.randn(m, spatial_dim), 
            requires_grad=False) for _ in range(num_spatial_features)
        ])
        
        # Initialize temporal Fourier feature matrices B_t (fixed during training)
        self.B_t = nn.ParameterList([
            nn.Parameter(sigma_t * torch.randn(m, 1),  # Temporal input is 1D (time)
            requires_grad=False) for _ in range(num_temporal_features)
        ])
        
        # Shared fully-connected network (applied to each embedding)
        self.shared_fc = nn.Sequential()
        input_size = 2 * m  # Each embedding is [cos(Bx), sin(Bx)] of size 2*m
        for _ in range(hidden_layers):
            self.shared_fc.append(nn.Linear(input_size, hidden_size))
            self.shared_fc.append(nn.Tanh())  # Using tanh activation as per paper
            input_size = hidden_size
        
        # Final linear layer to combine all multiplied features
        self.final_layer = nn.Linear(
            num_spatial_features * num_temporal_features * hidden_size, 
            output_size
        )
    
    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x (Tensor): Spatial coordinates (batch_size, spatial_dim).
            t (Tensor): Temporal coordinate (batch_size, 1).
            
        Returns:
            Tensor: Network output (batch_size, output_size).
        """
        batch_size = x.shape[0]
        
        # Compute all spatial embeddings: [cos(2πB_x^i x), sin(2πB_x^i x)]
        spatial_embs = []
        for B in self.B_x:
            proj = 2 * torch.pi * torch.matmul(x, B.T)  # (batch_size, m)
            emb = torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)  # (batch_size, 2m)
            spatial_embs.append(emb)
        # Stack: (batch_size, Mx, 2m)
        spatial_embs = torch.stack(spatial_embs, dim=1)
        
        # Compute all temporal embeddings: [cos(2πB_t^j t), sin(2πB_t^j t)]
        temporal_embs = []
        for B in self.B_t:
            proj = 2 * torch.pi * torch.matmul(t, B.T)  # (batch_size, m)
            emb = torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)  # (batch_size, 2m)
            temporal_embs.append(emb)
        # Stack: (batch_size, Mt, 2m)
        temporal_embs = torch.stack(temporal_embs, dim=1)
        
        # Process embeddings through shared FC network
        # Spatial: (batch_size*Mx, 2m) -> (batch_size*Mx, hidden_size)
        spatial_features = self.shared_fc(
            spatial_embs.view(-1, 2*self.m)
        ).view(batch_size, self.Mx, -1)  # (batch_size, Mx, hidden_size)
        
        # Temporal: (batch_size*Mt, 2m) -> (batch_size*Mt, hidden_size)
        temporal_features = self.shared_fc(
            temporal_embs.view(-1, 2*self.m)
        ).view(batch_size, self.Mt, -1)  # (batch_size, Mt, hidden_size)
        
        # Compute element-wise product for all (Mx, Mt) pairs
        spatial_features = spatial_features.unsqueeze(2)  # (batch, Mx, 1, hidden)
        temporal_features = temporal_features.unsqueeze(1)  # (batch, 1, Mt, hidden)
        multiplied = spatial_features * temporal_features  # (batch, Mx, Mt, hidden)
        
        # Flatten and pass through final layer
        out = self.final_layer(multiplied.flatten(start_dim=1))
        return out