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
import os
from typing import Any, Dict, List, Optional, Tuple, Sequence

import jax
from jax import numpy as jp
import numpy as np
import mujoco

from brax import math
from brax.envs import base
from brax.envs.base import State

from pupperv3_mjx.environment import PupperV3Env


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
    - Force estimator sees: full 36-dim frame with command/orientation slots zeroed out
    - Actor sees: IMU + estimator command + desired orientation + motors + action (36 dims)
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
        
        # Override observation dim to match original actor checkpoint (36 dims per frame)
        # Layout per frame: IMU(6) | velocity command(3) | desired orientation(3) | motors(12) | last action(12)
        self.observation_dim = 36
        
        # Auto-detect estimator frame count from loaded model's input dimension
        estimator_input_dim = self._force_estimator['input_mean'].shape[0]
        self._estimator_history = estimator_input_dim // self.observation_dim

        print(f"Loaded force estimator from {force_estimator_path}")
        print(f"Admittance gains: x={admittance_gains[0]}, y={admittance_gains[1]} m/s per N")
        print(f"Actor observation: 36 dims (estimator command + orientation + proprioception)")
        print(f"Force estimator history: {self._estimator_history} frames "
              f"({self._estimator_history * self.observation_dim} dims) [auto-detected]")
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
        
        # 1. Run force estimator on observation history (command/orientation slots masked to zero)
        estimated_force = self._estimate_force_from_observation(state)
        
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
    
    def _estimate_force_from_observation(self, state: State) -> jax.Array:
        """Mask command/orientation slots and run force estimator on 36-dim frames."""
        obs_frames = state.obs.reshape(self._observation_history, self.observation_dim)

        frames_to_use = min(self._estimator_history, self._observation_history)
        recent_frames = obs_frames[-frames_to_use:]

        def mask_frame(frame_obs):
            # Zero out command (indices 6:9) and orientation (indices 9:12) before feeding estimator
            frame_obs = frame_obs.at[6:9].set(0.0)
            frame_obs = frame_obs.at[9:12].set(0.0)
            return frame_obs

        masked_frames = jax.vmap(mask_frame)(recent_frames)
        estimator_input = masked_frames.reshape(-1)  # 36 dims × estimator_history

        return apply_force_estimator(estimator_input, self._force_estimator)
    
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
        
        Actor observation (36 dims per frame):
        - IMU data (angular velocity, gravity) - 6 dims
        - Velocity command from estimator - 3 dims
        - Desired orientation (world z in body frame) - 3 dims
        - Motor angles - 12 dims
        - Last action - 12 dims
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

        # Velocity command from estimator (actor tracks this)
        # Note: estimated_command is set in step() from force estimator output
        estimated_command = state_info.get('estimated_command', jp.zeros(3))

        # Desired orientation command (same as base environment)
        # Default to [0,0,1] = "stay level" to match pure RL training
        desired_orientation = state_info.get('desired_world_z_in_body_frame', jp.array([0.0, 0.0, 1.0]))

        # Construct observation WITH command from estimator
        obs = jp.concatenate(
            [
                lagged_imu_data,  # 6 dims: angular velocity + gravity
                estimated_command,  # 3 dims: velocity command from estimator
                desired_orientation,  # 3 dims: desired world z in body frame
                pipeline_state.q[7:] - self._default_pose + motor_ang_noise,  # 12 dims: motor angles
                state_info["last_act"] + last_action_noise,  # 12 dims: last action
            ]
        )
        # Total: 6 + 3 + 3 + 12 + 12 = 36 dims per frame

        obs = jp.clip(obs, -100.0, 100.0)

        # Stack observations through time
        new_obs_history = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return new_obs_history

    def render(
        self,
        trajectory: Sequence[base.State],
        camera: Optional[str] = None,
        force_vis_scale: float = 0.05,
    ) -> Sequence[np.ndarray]:
        """Render frames with both ground truth (red) and estimated (blue) forces."""
        if not trajectory:
            return []

        # If we were handed plain pipeline states (legacy path), fall back to base class.
        if not hasattr(trajectory[0], "pipeline_state"):
            return super().render(trajectory, camera=camera, force_vis_scale=force_vis_scale)

        camera = camera or "tracking_cam"
        width, height = 640, 480
        os.environ.setdefault("MUJOCO_GL", "egl")
        model = self.sys.mj_model
        renderer: Optional[mujoco.Renderer] = None
        gl_context: Optional[mujoco.GLContext] = None
        frames: List[np.ndarray] = []

        try:
            try:
                gl_context = mujoco.GLContext(max_width=width, max_height=height)
            except mujoco.FatalError:
                os.environ["MUJOCO_GL"] = "osmesa"
                gl_context = mujoco.GLContext(max_width=width, max_height=height)
            gl_context.make_current()
            renderer = mujoco.Renderer(model, height=height, width=width)

            for state in trajectory:
                pipeline_state = state.pipeline_state
                data = mujoco.MjData(model)
                mujoco.mj_resetData(model, data)
                data.qpos[:] = np.asarray(pipeline_state.q)
                data.qvel[:] = np.asarray(pipeline_state.qd)
                data.xfrc_applied[:] = np.asarray(pipeline_state.xfrc_applied)
                mujoco.mj_forward(model, data)

                renderer.update_scene(data, camera=camera)

                torso_pos = data.xpos[self._torso_body_id]
                origin = torso_pos + np.array([0.0, 0.0, 0.35])

                # Ground truth force (red)
                gt_force = np.asarray(data.xfrc_applied[self._torso_body_id, :3])
                if np.linalg.norm(gt_force) > 0.1:
                    self._draw_force_arrow(
                        renderer,
                        origin,
                        gt_force,
                        scale=force_vis_scale,
                        rgba=(1.0, 0.0, 0.0, 1.0),
                    )

                # Estimated force (blue)
                est_force = np.asarray(state.info.get("estimated_force", np.zeros(3)))
                if np.linalg.norm(est_force) > 0.1:
                    self._draw_force_arrow(
                        renderer,
                        origin,
                        est_force,
                        scale=force_vis_scale,
                        rgba=(0.0, 0.45, 1.0, 0.9),
                    )

                frames.append(renderer.render())
        finally:
            if renderer is not None:
                renderer.close()
            if gl_context is not None:
                gl_context.free()

        return frames

