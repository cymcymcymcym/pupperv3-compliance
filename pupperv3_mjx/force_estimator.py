"""Flax modules for force estimation."""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class ForceEstimator(nn.Module):
    """MLP that predicts 3D force vectors from observations."""

    hidden_size: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_size)(x)
        x = nn.elu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        x = nn.Dense(self.hidden_size)(x)
        x = nn.elu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        x = nn.Dense(3)(x)
        return x


class ForceEstimatorLarge(nn.Module):
    """Larger MLP with configurable layers for force estimation."""

    hidden_sizes: Sequence[int] = (512, 512, 256)
    dropout_rate: float = 0.1
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(hidden_size)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.elu(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        x = nn.Dense(3)(x)
        return x


class ForceEstimatorAutoregressive(nn.Module):
    """MLP that takes observation + previous force prediction as input.
    
    This encourages temporal smoothness by conditioning on past predictions.
    Input: [observation (720), previous_force (3)] = 723 dims
    Output: force (3)
    """

    hidden_sizes: Sequence[int] = (512, 512, 256)
    dropout_rate: float = 0.1
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, obs: jnp.ndarray, prev_force: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # Concatenate observation with previous force
        x = jnp.concatenate([obs, prev_force], axis=-1)
        
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.elu(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        x = nn.Dense(3)(x)
        return x


class ForceEstimatorGRU(nn.Module):
    """GRU-based force estimator for temporal modeling.
    
    Uses a GRU cell to maintain hidden state across timesteps.
    More powerful than autoregressive MLP but slower.
    """

    hidden_size: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, carry: jnp.ndarray, train: bool = False) -> tuple:
        """
        Args:
            obs: Current observation [batch, obs_dim]
            carry: GRU hidden state [batch, hidden_size]
            train: Whether in training mode
        Returns:
            (force_prediction, new_carry)
        """
        # GRU cell
        gru_cell = nn.GRUCell(features=self.hidden_size)
        new_carry, _ = gru_cell(carry, obs)
        
        # Output head
        x = nn.Dense(self.hidden_size)(new_carry)
        x = nn.elu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(3)(x)
        
        return x, new_carry






