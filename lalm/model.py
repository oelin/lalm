from typing import List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat


# Transformer modules.

def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMS norm."""

    rms = x.square().mean(dim=-1, keepdim=True).sqrt()

    return x / rms


class Attention(nn.Module):
    """Attention.

    Example
    -------
    >>> module = Attention(embedding_dimension=256, heads=16)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.heads = heads

        self.linear_1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
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

        x = self.linear_1(x)
        q, k, v = rearrange(x, 'b t (n h e) -> n b h t e', n=3, h=self.heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.linear_2(rearrange(x, 'b h t e -> b t (h e)'))

        return x


class MLP(nn.Module):
    """MLP.

    Example
    -------
    >>> module = MLP(embedding_dimension=256)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.linear_1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension * 3,
            out_features=embedding_dimension,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = F.silu(self.linear_1(x))
        x = F.silu(self.linear_2(x))

        return x


class TransformerBlock(nn.Module):
    """Transformer block.

    Example
    -------
    >>> module = TransformerBlock(embedding_dimension=256, heads=16)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

        self.mlp = MLP(embedding_dimension=embedding_dimension)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = x + self.attention(rms_norm(x))
        x = x + self.mlp(rms_norm(x))

        return x


@dataclass(frozen=True)
class TransformerConfiguration:
    embedding_dimension: int
    heads: int
    blocks: int


class Transformer(nn.Module):
    """Transformer.

    Example
    -------
    >>> configuration = TransformerConfiguration(
    ...     embedding_dimenson=256,
    ...     heads=16,
    ...     blocks=16,
    ... )
    >>> module = Transformer(configuration=configuration)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, configuration: TransformerConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : TransformerConfiguration
            The module configuration.
        """

        super().__init__()

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embedding_dimension=configuration.embedding_dimension,
                heads=configuration.heads,
            ) for _ in range(configuration.blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = self.blocks(x)

        return x


# LaLM modules.

class Downsample(nn.Module):
    """Downsample.

    Example
    -------
    >>> module = Downsample(embedding_dimension=256)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 5, 256).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.linear = nn.Linear(
            in_features=embedding_dimension * 2,
            out_features=embedding_dimension,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = rearrange(x, 'b (n t) e -> b t (n e)', n=2)
        x = self.linear(x)

        return x


class Upsample(nn.Module):
    """Upsample.

    Example
    -------
    >>> module = Upsample(embedding_dimension=256)
    >>> x = torch.randn((1, 5, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.linear = nn.Linear(
            in_features=embedding_dimension // 2,
            out_features=embedding_dimension,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = rearrange(x, 'b t (n e) -> b (n t) e', n=2)
        x = self.linear(x)

        return x


class DownBlock(nn.Module):
    """Down block.

    Example
    -------
    >>> module = DownBlock(embedding_dimension=256, heads=16)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 5, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()
        
        self.transformer = Transformer(
            configuration=TransformerConfiguration(
                embedding_dimension=embedding_dimension,
                heads=heads,
                blocks=2,
            ),
        )

        self.downsample = Downsample(embedding_dimension=embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = self.transformer(x)
        x = self.downsample(x)

        return x


class UpBlock(nn.Module):
    """Up block.

    Example
    -------
    >>> module = UpBlock(embedding_dimension=256, heads=16)
    >>> x = torch.randn((1, 5, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.transformer = Transformer(
            configuration=TransformerConfiguration(
                embedding_dimension=embedding_dimension,
                heads=heads,
                blocks=2,
            ),
        )

        self.upsample = Upsample(embedding_dimension=embedding_dimension)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = self.upsample(x)
        x = self.transformer(x)

        return x


@dataclass(frozen=True)
class LaLMConfiguration:
    embedding_dimension: int
    heads: int
    blocks: int
    latent_dimension: int
    latent_sequence_length: int


class LaLM(nn.Module):
    """LaLM.

    Example
    -------
    >>> configuration = LaLMConfiguration(
    ...     embedding_dimension=256,
    ...     heads=16,
    ...     blocks=16,
    ...     latent_dimension=16,
    ...     latent_sequence_length=100,
    ... )
    >>> module = LaLM(configuration=configuration)
    >>> x = torch.randn((1, 1000, 256))
    >>> z = module.encode(x)  # Shape: (1, 10, 16).
    >>> x = module.decode(z)  # Shape: (1, 1000, 256).
    """

    def __init__(self, *, configuration: LaLMConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : LaLMConfiguration
            The module configuration.
        """

        super().__init__()

        pass
