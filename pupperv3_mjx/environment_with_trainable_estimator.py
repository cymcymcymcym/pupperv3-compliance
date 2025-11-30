"""
PupperV3 Environment with Trainable Force Estimator.

This environment:
1. Runs force estimator inference (provided externally as a function)
2. Converts estimated force to velocity command via admittance
3. Stores (obs, ground_truth_force) pairs for supervised training
4. Actor trains with PPO, estimator trains with supervised learning

The force estimator is updated OUTSIDE the JAX-traced step function
to avoid tracer issues.
"""

import json
from typing import Any, Callable, Dict, Tuple

import jax
from jax import numpy as jp
import numpy as np

from pupperv3_mjx.environment import PupperV3Env
from brax.envs.base import State


class PupperV3EnvWithTrainableEstimator(PupperV3Env):
    """Environment that supports trainable force estimator.
    
    Key difference from frozen version:
    - Takes force_estimator_fn as a callable (can be updated between episodes)
    - Stores ground truth force for supervised learning
    """
    
    def __init__(
        self,
        force_estimator_fn: Callable[[jp.ndarray], jp.ndarray],
        admittance_gains: Tuple[float, float] = (0.5, 0.5),
        **kwargs
    ):
        """
        Args:
            force_estimator_fn: Function that takes obs and returns estimated force
            admittance_gains: (gain_x, gain_y) for force->velocity conversion
            **kwargs: All other args passed to PupperV3Env
        """
        super().__init__(**kwargs)
        self._force_estimator_fn = force_estimator_fn
        self._admittance_gains = jp.array(admittance_gains)
        
        print(f"Admittance gains: x={admittance_gains[0]}, y={admittance_gains[1]} m/s per N")
    
    def update_force_estimator(self, new_fn: Callable[[jp.ndarray], jp.ndarray]):
        """Update the force estimator function (called between training rounds)."""
        self._force_estimator_fn = new_fn
    
    def step(self, state: State, action: jp.ndarray) -> State:
        """Step with force estimation and data collection."""
        # Run force estimator
        estimated_force = self._force_estimator_fn(state.obs)
        
        # Convert to velocity command via admittance
        vel_x = self._admittance_gains[0] * estimated_force[0]
        vel_y = self._admittance_gains[1] * estimated_force[1]
        velocity_command = jp.array([vel_x, vel_y, 0.0])
        
        # Clip velocity command
        velocity_command = jp.clip(
            velocity_command,
            jp.array([self._linear_velocity_x_range[0], self._linear_velocity_y_range[0], -2.0]),
            jp.array([self._linear_velocity_x_range[1], self._linear_velocity_y_range[1], 2.0])
        )
        
        # Update state info (functional update)
        new_info = {**state.info}
        new_info['command'] = velocity_command
        new_info['estimated_force'] = estimated_force
        state = state.replace(info=new_info)
        
        # Run parent step (ground truth force is in state.info['force_current_vector'])
        next_state = super().step(state, action)
        
        return next_state
    
    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        new_info = {**state.info}
        new_info['estimated_force'] = jp.zeros(3)
        return state.replace(info=new_info)


def create_force_estimator_fn(params: Dict, input_mean: jp.ndarray, input_std: jp.ndarray):
    """Create a force estimator function from Flax params.
    
    Returns a JIT-compiled function that can be updated by passing new params.
    """
    
    def apply_estimator(obs: jp.ndarray) -> jp.ndarray:
        """Forward pass through force estimator."""
        x = (obs - input_mean) / input_std
        
        # Get sorted layer keys
        dense_keys = sorted([k for k in params.keys() if 'Dense' in k])
        ln_keys = sorted([k for k in params.keys() if 'LayerNorm' in k])
        
        num_hidden = len(ln_keys)
        
        for i in range(num_hidden):
            # Dense
            kernel = params[f'Dense_{i}']['kernel']
            bias = params[f'Dense_{i}']['bias']
            x = x @ kernel + bias
            
            # LayerNorm
            scale = params[f'LayerNorm_{i}']['scale']
            ln_bias = params[f'LayerNorm_{i}']['bias']
            mean = jp.mean(x, axis=-1, keepdims=True)
            var = jp.var(x, axis=-1, keepdims=True)
            x = (x - mean) / jp.sqrt(var + 1e-6)
            x = x * scale + ln_bias
            
            # ELU activation
            x = jp.where(x > 0, x, jp.exp(x) - 1)
        
        # Output layer
        kernel = params[f'Dense_{num_hidden}']['kernel']
        bias = params[f'Dense_{num_hidden}']['bias']
        x = x @ kernel + bias
        
        return x
    
    return jax.jit(apply_estimator)


def load_estimator_from_json(json_path: str) -> Tuple[Dict, jp.ndarray, jp.ndarray]:
    """Load force estimator params from JSON.
    
    Returns:
        (params_dict, input_mean, input_std)
    """
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    input_mean = jp.array(model_data['input_mean'])
    input_std = jp.array(model_data['input_std'])
    
    # Reconstruct params
    params = {}
    dense_idx = 0
    ln_idx = 0
    
    for layer in model_data['layers']:
        layer_type = layer.get('type', 'dense')
        weights = layer['weights']
        
        if layer_type == 'dense':
            params[f'Dense_{dense_idx}'] = {
                'kernel': jp.array(weights[0]),
                'bias': jp.array(weights[1])
            }
            dense_idx += 1
        elif layer_type == 'layer_norm':
            params[f'LayerNorm_{ln_idx}'] = {
                'scale': jp.array(weights[0]),
                'bias': jp.array(weights[1])
            }
            ln_idx += 1
    
    return params, input_mean, input_std


