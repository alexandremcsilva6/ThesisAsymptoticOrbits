"""
Neural Network Architecture for Approximating Solutions to Differential Equations.

This module provides neural network components for approximating solutions to ordinary differential equations (ODEs).

It defines:
1. 'LinearModel': A simple feedforward neural network for approximating a single function.
2. 'EquationsModel': A collection of 'LinearModels', where each network approximates a different function in a system of ODEs.
"""

from typing import List, Dict, Callable
import torch as th
from torch import nn

class LinearModel(nn.Module):
    """
    A feedforward neural network for function approximation.

    Args:
        features (List[int]): Number of neurons in each hidden layer.
        activation_function (Callable): Activation function used in hidden layers.
    """
    
    def __init__(self, features: List[int], activation_function: Callable[[th.Tensor], th.Tensor], *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        layers = []
        input_dim = 1  # Input is a scalar (e.g., time t)
        # Hidden layers
        for output_dim in features:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(activation_function)
            input_dim = output_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(input_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, t: th.Tensor) -> th.Tensor:
        """
        Forward pass through the network.

        Args:
            t (th.Tensor): Input tensor.
        
        Returns:
            th.Tensor: Predicted function value.
        """
        return self.layers(t)


class EquationsModel(nn.Module):
    """
    A collection of neural networks, where each network approximates a separate function in a system of ODEs.

    Args:
        functions (List[str]): Names of functions in the system (e.g., ['x', 'y']).
        features (List[int]): Number of neurons in each hidden layer.
        activation_function (Callable): Activation function used in hidden layers.
    """
    def __init__(self, functions: List[str], features: List[int], activation_function: Callable[[th.Tensor], th.Tensor],*args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Dictionary of neural networks, one per function
        self.equations = nn.ModuleDict({function: LinearModel(features, activation_function) for function in functions})

    def forward(self, t: th.Tensor) -> Dict[str, th.Tensor]:
        """
        Forward pass through all function networks.

        Args:
            t (th.Tensor): Input tensor.
        
        Returns:
            Dict[str, th.Tensor]: Predicted values for each function.
        """
        return {function: model(t) for function, model in self.equations.items()}
