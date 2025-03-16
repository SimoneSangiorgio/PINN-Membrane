import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from torch import Tensor


def sample_b(sigma: float, size: tuple, device = None) -> Tensor:
    r"""Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, \sigma^2)`

    Args:
        sigma (float): standard deviation
        size (tuple): size of the matrix sampled

    See :class:`~rff.layers.GaussianEncoding` for more details
    """
    device = device if device != None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.randn(size).to(device=device) * sigma


#@torch.jit.script
def gaussian_encoding(
        v: Tensor,
        b: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`

    See :class:`~rff.layers.GaussianEncoding` for more details.
    """
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class GaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features"""

    def __init__(self, sigma: Optional[float] = None,
                 input_size: Optional[float] = None,
                 encoded_size: Optional[float] = None,
                 b: Optional[Tensor] = None):
        r"""
        Args:
            sigma (Optional[float]): standard deviation
            input_size (Optional[float]): the number of input dimensions
            encoded_size (Optional[float]): the number of dimensions the `b` matrix maps to
            b (Optional[Tensor], optional): Optionally specify a :attr:`b` matrix already sampled
        Raises:
            ValueError:
                If :attr:`b` is provided and one of :attr:`sigma`, :attr:`input_size`,
                or :attr:`encoded_size` is provided. If :attr:`b` is not provided and one of
                :attr:`sigma`, :attr:`input_size`, or :attr:`encoded_size` is not provided.
        """
        super().__init__()
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
        if self.b == None:
            self.b = sample_b(self.sigma, (self.encoded_size, self.input_size))
        self.register_buffer('enc', self.b)
        model.layers[0] = self.encoded_size*2

    def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: Tensor mapping using random fourier features of shape :math:`(N, *, 2 \cdot \text{encoded_size})`
        """
        return gaussian_encoding(v, self.b)
    

def fourier_encoding(v: Tensor, B: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`"""
    gamma_v = 2 * torch.pi * torch.matmul(v, B.T) # B.T to ensure correct dimensions (N, encoded_size)
    return torch.cat([torch.cos(gamma_v), torch.sin(gamma_v)], dim=-1)


class FourierFeatureEncoding1(nn.Module):
     """Layer for mapping coordinates using random Fourier features"""

     def __init__(self, sigma: float, input_size: int, encoded_size: int):  # Required arguments, no optionals
        super().__init__()
        self.sigma = sigma
        self.encoded_size = encoded_size
        self.input_size = input_size

        self.b = sample_b(self.sigma, (self.encoded_size, self.input_size))
        self.enc = nn.Parameter(self.b)  # Use nn.Parameter for proper registration
        self.output_size = self.encoded_size * 2

     def setup(self, model):
        """
        Questo metodo viene chiamato per inizializzare l'encoding.
        Puoi usarlo per registrare buffer o parametri aggiuntivi.
        """
        # Inserisci qui il codice di inizializzazione, se necessario
        pass

     def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v})`"""
        return fourier_encoding(v, self.enc)
