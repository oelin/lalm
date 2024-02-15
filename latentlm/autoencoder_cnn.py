#@markdown Convolutional model.

from typing import Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class ResidualBlock(nn.Module):
    """Residual block.

    Example
    -------
    >>> module = ResidualBlock(embedding_dimension=256)
    >>> x = torch.randn((1, 256, 10))
    >>> x = module(x)  # Shape: (1, 256, 10).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dimension,
                out_channels=embedding_dimension,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.GroupNorm(
                num_groups=embedding_dimension // 8,
                num_channels=embedding_dimension,
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = x + self.layers(x)

        return x


class UpsampleBlock(nn.Module):
    """Upsample block.

    Example
    -------
    >>> module = UpsampleBlock(embedding_dimension=256)
    >>> x = torch.randn((1, 256, 5))
    >>> x = module(x)  # Shape: (1, 256, 10).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dimension // 2,
                out_channels=embedding_dimension,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.GroupNorm(
                num_groups=embedding_dimension // 8,
                num_channels=embedding_dimension,
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = rearrange(x, 'b (f e) s -> b e (f s)', f=2)
        x = self.layers(x)

        return x


class DownsampleBlock(nn.Module):
    """Downsample block.

    Example
    -------
    >>> module = DownsampleBlock(embedding_dimension=256)
    >>> x = torch.randn((1, 256, 10))
    >>> x = module(x)  # Shape: (1, 256, 5).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dimension * 2,
                out_channels=embedding_dimension,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.GroupNorm(
                num_groups=embedding_dimension // 8,
                num_channels=embedding_dimension,
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = rearrange(x, 'b e (f s) -> b (f e) s', f=2)
        x = self.layers(x)

        return x


class UpBlock(nn.Module):
    """Up block.

    Example
    -------
    >>> module = UpBlock(embedding_dimension=256)
    >>> x = torch.randn((1, 256, 10))
    >>> x = module(x)  # Shape: (1, 256, 10).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.layers = nn.Sequential(
            ResidualBlock(embedding_dimension=embedding_dimension),
            UpsampleBlock(embedding_dimension=embedding_dimension),
            ResidualBlock(embedding_dimension=embedding_dimension),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.layers(x)

        return x


class DownBlock(nn.Module):
    """Down block.

    Example
    -------
    >>> module = DownBlock(embedding_dimension=256)
    >>> x = torch.randn((1, 256, 10))
    >>> x = module(x)  # Shape: (1, 256, 5).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.layers = nn.Sequential(
            ResidualBlock(embedding_dimension=embedding_dimension),
            DownsampleBlock(embedding_dimension=embedding_dimension),
            ResidualBlock(embedding_dimension=embedding_dimension),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.layers(x)

        return x


class Quantizer(nn.Module):
    """Quantizer.

    Example
    -------
    >>> module = Quantizer(
    ...     embedding_dimension=256,
    ...     quantizer_dimension=4,
    ...     quantizer_bits=5,
    ... )
    >>> x = torch.randn((1, 256, 10))
    >>> x = module.encode(x)  # Shape: (1, 4, 10).
    >>> x = module.decode(x)  # Shape: (1, 256, 10).
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        quantizer_dimension: int,
        quantizer_bits: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        latent_dimension : int
            The latent dimension.
        vocabulary_size : int
            The vocabulary size.
        """

        super().__init__()

        self.scale = (2 ** quantizer_bits) // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dimension,
                out_channels=quantizer_dimension,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=quantizer_dimension,
                out_channels=embedding_dimension,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.GroupNorm(
                num_groups=embedding_dimension // 8,
                num_channels=embedding_dimension,
            ),
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.scale * self.encoder(x)
        x = x + (x.floor() - x).detach()

        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.decoder(x)

        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        z : torch.Tensor
            The output latent tensor.
        """

        z = self.encode(x)
        x = self.decode(z)

        return x, z


@dataclass(frozen=True)
class AutoencoderConfiguration:
    embedding_dimension: int
    quantizer_dimension: int
    quantizer_bits: int
    down_blocks: int


class Autoencoder(nn.Module):
    """Autoencoder.

    Example
    -------
    >>> configuration = AutoencoderConfiguration(
    ...     embedding_dimension=256,
    ...     quantizer_dimension=4,
    ...     quantizer_bits=5,
    ...     down_blocks=4,
    ... )
    >>> module = Autoencoder(configuration=configuration)
    >>> x = torch.randn((1, 256, 1024))
    >>> z = module.encode(x)  # Shape: (1, 4, 64).
    >>> x = module.decode(z)  # Shape: (1, 256, 1024).
    """

    def __init__(self, *, configuration: AutoencoderConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : AutoencoderConfiguration
            The module configuration.
        """

        super().__init__()

        self.encoder = nn.Sequential(*[
            DownBlock(embedding_dimension=configuration.embedding_dimension)
            for _ in range(configuration.down_blocks)
        ])

        self.decoder = nn.Sequential(*[
            UpBlock(embedding_dimension=configuration.embedding_dimension)
            for _ in range(configuration.down_blocks)
        ])

        self.quantizer = Quantizer(
            embedding_dimension=configuration.embedding_dimension,
            quantizer_dimension=configuration.quantizer_dimension,
            quantizer_bits=configuration.quantizer_bits,
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.encoder(x)
        x = self.quantizer.encode(x)

        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.quantizer.decode(x)
        x = self.decoder(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        z : torch.Tensor
            The output latent tensor.
        """

        z = self.encode(x)
        x = self.decode(z)

        return x, z 
