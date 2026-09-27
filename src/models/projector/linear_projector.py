import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """Multi-Layer Perceptron Projector for feature transformation.

    Args:
        input_dim: Dimension of the input features
        output_dim: Dimension of the output features
        inter_dim: Dimension of intermediate layers. Ignored when num_layers == 1.
            The structure becomes: input_dim -> inter_dim -> ... -> inter_dim -> output_dim
        num_layers: Number of linear layers (must be at least 1)
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inter_dim: int | None = None,
        num_layers: int = 1,
        bias: bool = False,
    ):
        super(MLPProjector, self).__init__()

        # Validate input parameters
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"input_dim and output_dim must be positive, got {input_dim} and {output_dim}")
        if num_layers > 1 and (inter_dim is None or inter_dim <= 0):
            raise ValueError(f"inter_dim must be positive when num_layers > 1, got {inter_dim}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inter_dim = inter_dim
        self.bias = bias
        self.num_layers = num_layers

        # Build network layers
        self.layers = self._build_layers()

        # Initialize weights
        self.apply(self._init_weights)

    def _build_layers(self) -> nn.Sequential:
        """Construct the sequential layers of the MLP projector"""
        layers = []
        mid_dim = self.inter_dim

        if self.num_layers == 1:
            layers.append(nn.Linear(self.input_dim, self.output_dim, bias=self.bias))
        else:
            # First layer: input_dim -> mid_dim
            layers.append(nn.Linear(self.input_dim, mid_dim, bias=self.bias))
            # Intermediate layers: mid_dim -> mid_dim
            for _ in range(1, self.num_layers - 1):
                layers.extend([nn.GELU(approximate="tanh"), nn.Linear(mid_dim, mid_dim, bias=self.bias)])
            # Last layer: mid_dim -> output_dim
            layers.extend([nn.GELU(approximate="tanh"), nn.Linear(mid_dim, self.output_dim, bias=self.bias)])

        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for different module types"""
        init_std = 0.02  # Standard deviation for weight initialization

        if isinstance(module, nn.Linear):
            # Initialize linear layer weights with normal distribution
            module.weight.data.normal_(mean=0.0, std=init_std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights
            module.weight.data.normal_(mean=0.0, std=init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
