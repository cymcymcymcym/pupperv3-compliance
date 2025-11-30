"""
PupperV3 Environment with integrated Force Estimator and Admittance Controller.

This environment extends PupperV3Env to:
1. Run a pretrained force estimator on observations
2. Convert estimated force to velocity command via admittance
3. Use the velocity command as the tracking target for the actor

The actor learns to track velocity commands derived from (potentially noisy)
force estimates, making it robust to estimator errors.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import jax
from jax import numpy as jp
import numpy as np

from pupperv3_mjx.environment import PupperV3Env
from brax.envs.base import State


def load_force_estimator(json_path: str) -> Dict[str, Any]:
    """Load force estimator from exported JSON file.
    
    Returns dict with:
        - input_mean: normalization mean
        - input_std: normalization std  
        - layers: list of layer configs with weights
    """
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    # Convert lists to JAX arrays for efficient computation
    model_data['input_mean'] = jp.array(model_data['input_mean'])
    model_data['input_std'] = jp.array(model_data['input_std'])
    
    for layer in model_data['layers']:
        layer['weights'] = [jp.array(w) for w in layer['weights']]
    
    return model_data


def apply_force_estimator(obs: jax.Array, model: Dict[str, Any]) -> jax.Array:
    """Run forward pass through force estimator.
    
    Args:
        obs: Observation vector [obs_dim] or [batch, obs_dim]
        model: Loaded model dict from load_force_estimator
        
    Returns:
        Estimated force vector [3] or [batch, 3]
    """
    # Normalize input
    x = (obs - model['input_mean']) / model['input_std']
    
    # Forward pass through layers
    for layer in model['layers']:
        layer_type = layer.get('type', 'dense')
        
        if layer_type == 'dense':
            kernel, bias = layer['weights']
            x = x @ kernel + bias
        elif layer_type == 'layer_norm':
            scale, bias = layer['weights']
            # LayerNorm: normalize then scale and shift
            mean = jp.mean(x, axis=-1, keepdims=True)
            var = jp.var(x, axis=-1, keepdims=True)
            x = (x - mean) / jp.sqrt(var + 1e-6)
            x = x * scale + bias
        
        # Apply activation
        activation = layer.get('activation', 'identity')
        if activation == 'elu':
            x = jp.where(x > 0, x, jp.exp(x) - 1)
        elif activation == 'relu':
            x = jp.maximum(0, x)
        elif activation == 'tanh':
            x = jp.tanh(x)
        # identity: no-op
    
    return x


class PupperV3EnvWithEstimator(PupperV3Env):
    """PupperV3 Environment with integrated force estimator and admittance controller.
    
    This environment:
    1. Runs a pretrained force estimator on the current observation
    2. Converts estimated force to velocity command via admittance: vel = gain * force
    3. Uses the velocity command as the tracking target (replaces random command)
    
    The actor learns to track velocity commands derived from force estimates,
    making it robust to estimator noise.
    """
    
    def __init__(
        self,
        force_estimator_path: str,
        admittance_gains: Tuple[float, float] = (0.5, 0.5),
        use_ground_truth_force: bool = False,
        **kwargs
    ):
        """
        Args:
            force_estimator_path: Path to exported force_estimator.json
            admittance_gains: (gain_x, gain_y) for force->velocity conversion [m/s per N]
            use_ground_truth_force: If True, use actual force instead of estimated (for debugging)
            **kwargs: All other args passed to PupperV3Env
        """
        super().__init__(**kwargs)
        
        # Load force estimator
        self._force_estimator = load_force_estimator(force_estimator_path)
        self._admittance_gains = jp.array(admittance_gains)
        self._use_ground_truth_force = use_ground_truth_force
        
        print(f"Loaded force estimator from {force_estimator_path}")
        print(f"Admittance gains: x={admittance_gains[0]}, y={admittance_gains[1]} m/s per N")
    
    def _estimate_force(self, obs: jax.Array) -> jax.Array:
        """Run force estimator on observation.
        
        Args:
            obs: Full observation history [obs_dim * history]
            
        Returns:
            Estimated force [3]
        """
        return apply_force_estimator(obs, self._force_estimator)
    
    def _force_to_velocity_command(self, force: jax.Array) -> jax.Array:
        """Convert force to velocity command via admittance.
        
        Args:
            force: Force vector [fx, fy, fz]
            
        Returns:
            Command vector [vx, vy, ang_vel] where ang_vel=0
        """
        vel_x = self._admittance_gains[0] * force[0]
        vel_y = self._admittance_gains[1] * force[1]
        # No angular velocity from force (could add torque-based if needed)
        return jp.array([vel_x, vel_y, 0.0])
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step the environment with force estimator in the loop.
        
        1. Run force estimator on current observation
        2. Convert to velocity command via admittance
        3. Set as tracking target
        4. Run parent step (which computes tracking_lin_vel reward)
        """
        # Get force (estimated or ground truth)
        if self._use_ground_truth_force:
            force = state.info['force_current_vector']
        else:
            force = self._estimate_force(state.obs)
        
        # Convert force to velocity command
        velocity_command = self._force_to_velocity_command(force)
        
        # Clip velocity command to reasonable range
        velocity_command = jp.clip(
            velocity_command,
            jp.array([self._linear_velocity_x_range[0], self._linear_velocity_y_range[0], -2.0]),
            jp.array([self._linear_velocity_x_range[1], self._linear_velocity_y_range[1], 2.0])
        )
        
        # Override the command with admittance-derived velocity
        state.info['command'] = velocity_command
        
        # Store estimated force for logging/debugging
        state.info['estimated_force'] = force
        
        # Call parent step (uses tracking_lin_vel reward with our velocity command)
        return super().step(state, action)
    
    def reset(self, rng: jax.Array) -> State:
        """Reset with additional info fields for force estimation."""
        state = super().reset(rng)
        
        # Initialize estimated force field
        state.info['estimated_force'] = jp.zeros(3)
        
        # Set initial command to zero (will be overwritten in first step)
        state.info['command'] = jp.zeros(3)
        
        return state

