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

from brax import math
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
    1. Runs a pretrained force estimator on observations (WITHOUT command to avoid circular dep)
    2. Converts estimated force to velocity command via admittance: vel = gain * force
    3. Gives velocity command to actor as part of observation (actor sees command from estimator)
    4. Reward is based on tracking GROUND TRUTH velocity (not estimated)
    
    Key insight:
    - Force estimator sees: IMU + motors + action (30 dims) - NO command
    - Actor sees: IMU + motors + action + COMMAND FROM ESTIMATOR (33 dims)
    - Reward tracks: Ground truth velocity (from ground truth force)
    """
    
    def __init__(
        self,
        force_estimator_path: str,
        admittance_gains: Tuple[float, float] = (0.5, 0.5),
        **kwargs
    ):
        """
        Args:
            force_estimator_path: Path to exported force_estimator.json
            admittance_gains: (gain_x, gain_y) for force->velocity conversion [m/s per N]
            **kwargs: All other args passed to PupperV3Env
        """
        super().__init__(**kwargs)
        
        # Load force estimator
        self._force_estimator = load_force_estimator(force_estimator_path)
        self._admittance_gains = jp.array(admittance_gains)
        
        # Override observation dim: base (30) + command (3) = 33
        self.observation_dim = 33
        
        print(f"Loaded force estimator from {force_estimator_path}")
        print(f"Admittance gains: x={admittance_gains[0]}, y={admittance_gains[1]} m/s per N")
        print(f"Actor observation: 33 dims (includes velocity command from estimator)")
        print(f"Reward uses GROUND TRUTH force (not estimated)")
    
    def _estimate_force(self, obs: jax.Array) -> jax.Array:
        """Run force estimator on observation.
        
        Args:
            obs: Full observation history [obs_dim * history]
            
        Returns:
            Estimated force [3]
        """
        return apply_force_estimator(obs, self._force_estimator)
    
    def _force_to_velocity_command(self, force_world: jax.Array, body_rotation: jax.Array) -> jax.Array:
        """Convert world-frame force to body-frame velocity command via admittance.
        
        The force is applied in world frame, but velocity tracking reward uses body frame.
        So we need to rotate the force into body frame before computing velocity command.
        
        Args:
            force_world: Force vector in WORLD frame [fx, fy, fz]
            body_rotation: Body quaternion (world to body rotation)
            
        Returns:
            Command vector [vx, vy, ang_vel] in BODY frame where ang_vel=0
        """
        # Rotate world-frame force into body frame
        # quat_inv gives the inverse rotation (world -> body)
        force_body = math.rotate(force_world, math.quat_inv(body_rotation))
        
        # Apply admittance gains to body-frame force
        vel_x = self._admittance_gains[0] * force_body[0]
        vel_y = self._admittance_gains[1] * force_body[1]
        # No angular velocity from force (could add torque-based if needed)
        return jp.array([vel_x, vel_y, 0.0])
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step the environment with force estimator in the loop.
        
        Flow:
        1. Run force estimator on current obs (WITHOUT command) → estimated force
        2. Convert estimated force → estimated velocity command (for ACTOR observation)
        3. Compute ground truth velocity command (for REWARD)
        4. Run parent step
        
        Key design:
        - Actor SEES: velocity command from ESTIMATOR (in observation)
        - Actor is REWARDED for: tracking GROUND TRUTH velocity
        - This trains actor to track commands, while reward ensures correct behavior
        """
        # Get body rotation for frame conversion
        body_rotation = state.pipeline_state.x.rot[self._torso_idx]
        
        # 1. Run force estimator on observation (obs doesn't include command yet)
        #    Note: state.obs here is from previous step, which has estimated_command from prev step
        #    We need to extract the base observation (without command) for the force estimator
        #    The force estimator was trained on 30-dim obs (no command)
        estimated_force = self._estimate_force_from_base_obs(state)
        
        # 2. Convert estimated force → velocity command (this goes into ACTOR's observation)
        estimated_command = self._force_to_velocity_command(estimated_force, body_rotation)
        estimated_command = jp.clip(
            estimated_command,
            jp.array([self._linear_velocity_x_range[0], self._linear_velocity_y_range[0], -2.0]),
            jp.array([self._linear_velocity_x_range[1], self._linear_velocity_y_range[1], 2.0])
        )
        
        # 3. Compute GROUND TRUTH velocity command (for REWARD)
        ground_truth_force = state.info['force_current_vector']
        ground_truth_command = self._force_to_velocity_command(ground_truth_force, body_rotation)
        ground_truth_command = jp.clip(
            ground_truth_command,
            jp.array([self._linear_velocity_x_range[0], self._linear_velocity_y_range[0], -2.0]),
            jp.array([self._linear_velocity_x_range[1], self._linear_velocity_y_range[1], 2.0])
        )
        
        # Update state.info
        new_info = {**state.info}
        new_info['command'] = ground_truth_command  # For REWARD (tracking_lin_vel uses this)
        new_info['estimated_command'] = estimated_command  # For ACTOR observation (next step)
        new_info['estimated_force'] = estimated_force  # For logging
        
        # Replace state with updated info
        state = state.replace(info=new_info)
        
        # Call parent step (reward uses ground_truth_command via state.info['command'])
        return super().step(state, action)
    
    def _estimate_force_from_base_obs(self, state: State) -> jax.Array:
        """Extract base observation (without command) and run force estimator.
        
        The force estimator was trained on 30-dim observations:
        - IMU (6) + motor angles (12) + last action (12) = 30 dims per frame
        
        But actor observation is 33 dims (includes command).
        We need to extract just the base 30 dims for the estimator.
        """
        # state.obs is [obs_dim * history] = [33 * 20] = 660 for actor
        # We need to extract [30 * 20] = 600 for force estimator
        
        # Each frame is 33 dims: [imu(6), command(3), motors(12), action(12)]
        # Force estimator expects: [imu(6), motors(12), action(12)] = 30 dims
        
        obs_history = self._observation_history
        actor_frame_dim = 33  # IMU(6) + cmd(3) + motors(12) + action(12)
        estimator_frame_dim = 30  # IMU(6) + motors(12) + action(12)
        
        # Extract base observation for each frame
        def extract_base_frame(frame_obs):
            # frame_obs is [33]: [imu(6), cmd(3), motors(12), action(12)]
            # Return [30]: [imu(6), motors(12), action(12)]
            imu = frame_obs[:6]
            motors = frame_obs[9:21]  # Skip command (indices 6-8)
            action = frame_obs[21:33]
            return jp.concatenate([imu, motors, action])
        
        # Reshape to [history, frame_dim], extract, reshape back
        obs_frames = state.obs.reshape(obs_history, actor_frame_dim)
        base_frames = jax.vmap(extract_base_frame)(obs_frames)
        base_obs = base_frames.reshape(-1)  # [30 * 20] = 600
        
        return apply_force_estimator(base_obs, self._force_estimator)
    
    def reset(self, rng: jax.Array) -> State:
        """Reset with additional info fields for force estimation."""
        state = super().reset(rng)
        
        # Update info using functional update (JAX-compatible)
        new_info = {**state.info}
        new_info['estimated_force'] = jp.zeros(3)
        new_info['command'] = jp.zeros(3)
        new_info['estimated_command'] = jp.zeros(3)  # Command from estimator (for actor obs)
        
        return state.replace(info=new_info)
    
    def _get_obs(
        self,
        pipeline_state,
        state_info: dict,
        obs_history: jax.Array,
    ) -> jax.Array:
        """Get observation for ACTOR - includes velocity command from estimator.
        
        Actor observation (33 dims per frame):
        - IMU data (angular velocity, gravity) - 6 dims
        - Motor angles - 12 dims
        - Last action - 12 dims
        - Velocity command from estimator - 3 dims  <-- This is what actor needs to track!
        """
        from pupperv3_mjx import utils
        
        if self._use_imu:
            inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
            local_body_angular_velocity = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        else:
            inv_torso_rot = jp.array([1, 0, 0, 0])
            local_body_angular_velocity = jp.zeros(3)

        # Noise keys
        (
            state_info["rng"],
            ang_key,
            gravity_key,
            motor_angle_key,
            last_action_key,
            imu_sample_key,
        ) = jax.random.split(state_info["rng"], 6)

        ang_vel_noise = (
            jax.random.uniform(ang_key, (3,), minval=-1, maxval=1) * self._angular_velocity_noise
        )
        gravity_noise = (
            jax.random.uniform(gravity_key, (3,), minval=-1, maxval=1) * self._gravity_noise
        )
        motor_ang_noise = (
            jax.random.uniform(motor_angle_key, (12,), minval=-1, maxval=1)
            * self._motor_angle_noise
        )
        last_action_noise = (
            jax.random.uniform(last_action_key, (12,), minval=-1, maxval=1)
            * self._last_action_noise
        )

        noised_gravity = math.rotate(jp.array([0, 0, -1]), inv_torso_rot) + gravity_noise
        noised_gravity = noised_gravity / (jp.linalg.norm(noised_gravity) + 1e-6)
        noised_ang_vel = local_body_angular_velocity + ang_vel_noise
        noised_imu_data = jp.concatenate([noised_ang_vel, noised_gravity])

        lagged_imu_data, state_info["imu_buffer"] = utils.sample_lagged_value(
            imu_sample_key,
            state_info["imu_buffer"],
            noised_imu_data,
            self._imu_latency_distribution,
        )

        # Get velocity command from estimator (this is what actor should track)
        # Note: estimated_command is set in step() from force estimator output
        estimated_command = state_info.get('estimated_command', jp.zeros(3))

        # Construct observation WITH command from estimator
        obs = jp.concatenate(
            [
                lagged_imu_data,  # 6 dims: angular velocity + gravity
                estimated_command,  # 3 dims: velocity command FROM ESTIMATOR (actor tracks this!)
                pipeline_state.q[7:] - self._default_pose + motor_ang_noise,  # 12 dims: motor angles
                state_info["last_act"] + last_action_noise,  # 12 dims: last action
            ]
        )
        # Total: 6 + 3 + 12 + 12 = 33 dims per frame

        obs = jp.clip(obs, -100.0, 100.0)

        # Stack observations through time
        new_obs_history = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return new_obs_history

