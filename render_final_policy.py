#!/usr/bin/env python3
"""
Render the final compliance policy with both ground truth and estimated force arrows.

Shows:
- Red arrow: Actual applied force (ground truth)
- Blue arrow: Estimated force (from force estimator)
- Green arrow: Velocity command derived from admittance

Usage:
    python render_final_policy.py \
        --force-estimator-path final_force_estimator.json \
        --actor-checkpoint-path final_actor_checkpoint/output_joint_supervised_2025-12-01_11-12-03/round_9/actor_checkpoint
"""

import argparse
import json
import os
import functools
from pathlib import Path
from typing import Tuple, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from jax import numpy as jp
import mediapy as media
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from orbax import checkpoint as ocp
from ml_collections import config_dict

from brax.io import mjcf
from brax.training.agents.ppo import networks as ppo_networks

from pupperv3_mjx import config, environment, utils
from pupperv3_mjx.environment_with_estimator import PupperV3EnvWithEstimator


# ============================================================================
# Force Estimator Observation Processing
# ============================================================================

ESTIMATOR_FRAMES = 10  # Force estimator uses last 10 frames


def load_force_estimator(estimator_path: Path, estimator_frames: int = ESTIMATOR_FRAMES):
    """Load force estimator from JSON file.
    
    Args:
        estimator_path: Path to force_estimator.json
        estimator_frames: Number of frames the estimator expects (default 10)
    
    Returns:
        Tuple of (estimator_fn, expected_input_dim)
    """
    with open(estimator_path) as f:
        est_dict = json.load(f)

    layers = est_dict["layers"]
    input_mean = jp.array(est_dict["input_mean"])
    input_std = jp.array(est_dict["input_std"])
    
    # Infer expected input dimension from first layer
    first_dense = next(l for l in layers if l.get("type", "dense") == "dense")
    expected_input_dim = len(first_dense["weights"][0])
    
    print(f"  Force estimator expects {expected_input_dim} input dims")
    print(f"  Input mean shape: {input_mean.shape}, std shape: {input_std.shape}")

    parsed_layers = []
    for layer in layers:
        layer_type = layer.get("type", "dense")
        weights = layer["weights"]
        activation = layer.get("activation", "identity")
        if layer_type == "dense":
            kernel = jp.array(weights[0])
            bias = jp.array(weights[1])
            parsed_layers.append(("dense", kernel, bias, activation))
        elif layer_type == "layer_norm":
            scale = jp.array(weights[0])
            bias = jp.array(weights[1])
            parsed_layers.append(("layer_norm", scale, bias, activation))

    def apply_activation(x, activation):
        if activation == "elu":
            return jax.nn.elu(x)
        elif activation == "tanh":
            return jp.tanh(x)
        elif activation == "identity" or activation is None:
            return x
        else:
            return x

    def estimator_fn(obs: jp.ndarray) -> jp.ndarray:
        """Apply force estimator to preprocessed observation."""
        # Handle zero std
        input_std_safe = jp.where(input_std < 1e-6, 1.0, input_std)
        x = (obs - input_mean) / input_std_safe
        
        for layer_info in parsed_layers:
            layer_type = layer_info[0]
            if layer_type == "dense":
                _, kernel, bias, activation = layer_info
                x = x @ kernel + bias
                x = apply_activation(x, activation)
            elif layer_type == "layer_norm":
                _, scale, bias, activation = layer_info
                mean = jp.mean(x, axis=-1, keepdims=True)
                var = jp.var(x, axis=-1, keepdims=True)
                x = (x - mean) / jp.sqrt(var + 1e-6)
                x = x * scale + bias
                x = apply_activation(x, activation)
        return x

    return estimator_fn, expected_input_dim


def prepare_estimator_input(
    obs: jp.ndarray, 
    env_frame_dim: int = 30,
    estimator_frame_dim: int = 36,
    observation_history: int = 20,
    estimator_frames: int = ESTIMATOR_FRAMES,
) -> jp.ndarray:
    """Convert environment observation to force estimator input format.
    
    The base environment (PupperV3Env) has 30-dim frames:
        - IMU data: 6 dims (indices 0-5)
        - Motor angles: 12 dims (indices 6-17)
        - Last action: 12 dims (indices 18-29)
    
    The force estimator was trained on 36-dim frames:
        - IMU data: 6 dims (indices 0-5)
        - Command: 3 dims (indices 6-8) - MASKED TO ZERO
        - Orientation: 3 dims (indices 9-11) - MASKED TO ZERO
        - Motor angles: 12 dims (indices 12-23)
        - Last action: 12 dims (indices 24-35)
    
    Args:
        obs: Raw observation from environment [env_frame_dim * observation_history]
        env_frame_dim: Dimension per frame from environment (30 for base env)
        estimator_frame_dim: Dimension per frame expected by estimator (36)
        observation_history: Total frames in observation (20)
        estimator_frames: Number of frames to use for estimator (10)
    
    Returns:
        Processed observation for force estimator [estimator_frame_dim * estimator_frames]
    """
    # Reshape to frames
    total_obs_dim = env_frame_dim * observation_history
    if obs.shape[-1] != total_obs_dim:
        # Maybe already 36-dim frames?
        if obs.shape[-1] == estimator_frame_dim * observation_history:
            # Already in 36-dim format, just extract and mask
            frames = obs.reshape(observation_history, estimator_frame_dim)
            recent_frames = frames[-estimator_frames:]
            # Mask command/orientation (indices 6:12)
            masked_frames = recent_frames.at[:, 6:12].set(0.0)
            return masked_frames.reshape(-1)
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}, expected {total_obs_dim} or {estimator_frame_dim * observation_history}")
    
    frames = obs.reshape(observation_history, env_frame_dim)
    
    # Take only the most recent frames
    recent_frames = frames[-estimator_frames:]
    
    # Convert 30-dim to 36-dim format by inserting 6 zeros for command/orientation
    # 30-dim: [IMU(6), motors(12), action(12)]
    # 36-dim: [IMU(6), cmd(3), orient(3), motors(12), action(12)]
    def expand_frame(frame_30):
        imu = frame_30[:6]
        motors = frame_30[6:18]
        action = frame_30[18:30]
        cmd_orient = jp.zeros(6)  # Masked command + orientation
        return jp.concatenate([imu, cmd_orient, motors, action])
    
    expanded_frames = jax.vmap(expand_frame)(recent_frames)
    
    return expanded_frames.reshape(-1)


def draw_force_arrow(
    renderer: mujoco.Renderer,
    origin: np.ndarray,
    force: np.ndarray,
    scale: float = 0.05,
    rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
) -> None:
    """Draw a force arrow into the MuJoCo renderer scene."""
    p1 = np.asarray(origin, dtype=np.float64)
    p2 = p1 + np.asarray(force, dtype=np.float64) * scale
    geom_index = renderer.scene.ngeom
    if geom_index >= renderer.scene.maxgeom:
        return
    geom = renderer.scene.geoms[geom_index]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.01, p1, p2)
    renderer.scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description="Render final compliance policy")
    parser.add_argument(
        "--force-estimator-path",
        type=str,
        default="final_force_estimator.json",
        help="Path to force_estimator.json"
    )
    parser.add_argument(
        "--actor-checkpoint-path",
        type=str,
        default="final_actor_checkpoint/output_joint_supervised_2025-12-01_11-12-03/round_9/actor_checkpoint",
        help="Path to actor Orbax checkpoint"
    )
    parser.add_argument(
        "--admittance-gains",
        type=str,
        default="0.1,0.1",
        help="Admittance gains (x,y)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations.xml",
        help="Path to MuJoCo XML model"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=600,
        help="Number of steps to simulate"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="videos/final_compliance_demo.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use-noise",
        action="store_true",
        help="Replace estimator output with random noise (ablation test)"
    )
    parser.add_argument(
        "--use-zero",
        action="store_true",
        help="Replace estimator output with zeros (ablation test)"
    )
    parser.add_argument(
        "--force-magnitude-range",
        type=str,
        default="2,6",
        help="Force magnitude range (min,max) in Newtons for noise generation"
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Random seed for noise generation (default: uses --seed)"
    )
    
    args = parser.parse_args()
    
    admittance_gains = tuple(float(x) for x in args.admittance_gains.split(","))
    force_magnitude_range = tuple(float(x) for x in args.force_magnitude_range.split(","))
    
    print(f"Loading force estimator from: {args.force_estimator_path}")
    print(f"Loading actor from: {args.actor_checkpoint_path}")
    print(f"Admittance gains: {admittance_gains}")
    
    if args.use_noise and args.use_zero:
        print("ERROR: Cannot use both --use-noise and --use-zero. Pick one.")
        return
    
    if args.use_noise:
        # Noise magnitude = force_magnitude * 0.2, so range is [min*0.2, max*0.2]
        noise_force_min = force_magnitude_range[0] * 0.2
        noise_force_max = force_magnitude_range[1] * 0.2
        print(f"⚠️  ABLATION MODE: Using random NOISE instead of estimator!")
        print(f"   Noise force magnitude range: [{noise_force_min:.2f}, {noise_force_max:.2f}] N")
    
    if args.use_zero:
        print(f"⚠️  ABLATION MODE: Using ZERO force instead of estimator!")
    
    # Load force estimator
    estimator_fn, expected_input_dim = load_force_estimator(Path(args.force_estimator_path))
    jit_estimator = jax.jit(estimator_fn)
    
    # Determine estimator frame count from input dimension
    # 36 dims per frame, so input_dim / 36 = num_frames
    estimator_frames = expected_input_dim // 36
    print(f"  Force estimator uses {estimator_frames} frames ({expected_input_dim} dims)")
    
    # Setup environment config
    reward_config = config.get_config()
    reward_config.rewards.scales.tracking_lin_vel = 1.5
    reward_config.rewards.scales.tracking_ang_vel = 0.0
    reward_config.rewards.scales.force_following = 0.0
    
    # Environment parameters (must match training)
    observation_history = 20
    action_scale = 0.3
    kp = 30.0
    kd = 1.0
    
    # Use PupperV3EnvWithEstimator to get 36-dim frames (720-dim obs) matching actor training
    env = PupperV3EnvWithEstimator(
        path=args.model_path,
        reward_config=reward_config,
        action_scale=action_scale,
        observation_history=observation_history,
        dof_damping=kd,
        position_control_kp=kp,
        force_probability=0.8,
        force_duration_range=jp.array([40, 120]),
        force_magnitude_range=jp.array([2, 6]),  # Must match training!
        force_estimator_path=args.force_estimator_path,
        admittance_gains=admittance_gains,
    )
    
    print(f"  Environment observation size: {env.observation_size} (expecting 720)")
    
    # Load actor from Orbax checkpoint
    print("Loading actor checkpoint...")
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(args.actor_checkpoint_path)
    
    # restored is [normalizer_params, network_params]
    normalizer_params = restored[0]
    network_params = restored[1]
    
    print(f"  Normalizer keys: {list(normalizer_params.keys())}")
    print(f"  Network keys: {list(network_params.keys())}")
    
    # Create policy network - use obs size from normalizer (actor checkpoint) 
    # to ensure compatibility
    obs_size_from_normalizer = normalizer_params['mean'].shape[0]
    obs_size = env.observation_size
    action_size = env.action_size
    
    print(f"  Env obs size: {obs_size}, Actor normalizer obs size: {obs_size_from_normalizer}")
    
    if obs_size != obs_size_from_normalizer:
        print(f"  WARNING: Obs size mismatch! Using normalizer size {obs_size_from_normalizer}")
        obs_size = obs_size_from_normalizer
    
    # Network architecture (must match training)
    policy_hidden_sizes = (256, 128, 128, 128)
    
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=lambda x, y: x,  # We'll handle normalization ourselves
        policy_hidden_layer_sizes=policy_hidden_sizes,
        activation=jax.nn.elu,
    )
    
    # Create inference function using Brax's make_inference_fn pattern
    # The network apply signature is: apply(processor_params, policy_params, obs)
    # processor_params is for normalization, policy_params is the actual network weights
    
    # Rebuild networks - use identity preprocessing since we'll normalize manually
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=lambda x, y: x,  # Identity - we normalize in policy_fn
        policy_hidden_layer_sizes=policy_hidden_sizes,
        activation=jax.nn.elu,
    )
    
    def policy_fn(obs, rng):
        # Normalize observation
        obs_mean = normalizer_params['mean']
        obs_std = normalizer_params['std']
        obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
        
        # Apply policy network
        # Signature: apply(processor_params, policy_params, obs)
        # processor_params is empty dict since we use identity preprocessing
        raw_action = ppo_network.policy_network.apply({}, network_params['policy'], obs_norm)
        
        # For deterministic, just use the mean (first action_size elements)
        # raw_action shape is (2 * action_size,) for Gaussian: [mean, log_std]
        action = raw_action[:action_size]
        return jp.tanh(action), {}
    jit_policy = jax.jit(policy_fn)
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # Run simulation
    rng = jax.random.PRNGKey(34)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    
    rollout = [state.pipeline_state]
    actual_forces = []
    estimated_forces = []
    velocity_commands = []
    
    # EMA smoothing for predictions (match training)
    smoothed_prediction = np.zeros(3)
    prediction_smoothing = 0.5
    
    print(f"Running {args.num_steps} steps...")
    
    # JIT the observation preprocessing
    jit_prepare_input = jax.jit(
        lambda obs: prepare_estimator_input(
            obs, 
            env_frame_dim=env.observation_dim,
            estimator_frame_dim=36,
            observation_history=observation_history,
            estimator_frames=estimator_frames
        )
    )
    
    # Setup noise generator if using noise ablation
    noise_seed = args.noise_seed if args.noise_seed is not None else args.seed
    noise_rng = np.random.default_rng(noise_seed)
    if args.use_noise:
        print(f"   Noise seed: {noise_seed}")
    
    for step in range(args.num_steps):
        rng, act_rng = jax.random.split(rng)
        
        if args.use_zero:
            # ABLATION: Replace estimator with zeros
            raw_prediction = np.zeros(3)
        elif args.use_noise:
            # ABLATION: Replace estimator with random noise
            # Generate random direction (unit vector)
            noise_direction = noise_rng.standard_normal(3)
            noise_direction = noise_direction / (np.linalg.norm(noise_direction) + 1e-6)
            # Generate random magnitude in range [force_min*0.2, force_max*0.2]
            noise_force_min = force_magnitude_range[0] * 0.2
            noise_force_max = force_magnitude_range[1] * 0.2
            noise_magnitude = noise_rng.uniform(noise_force_min, noise_force_max)
            raw_prediction = noise_direction * noise_magnitude
        else:
            # Normal mode: Use force estimator
            # Prepare observation for force estimator (extract last N frames, mask, reshape)
            estimator_input = jit_prepare_input(state.obs)
            raw_prediction = np.asarray(jit_estimator(estimator_input))
        
        smoothed_prediction = prediction_smoothing * raw_prediction + (1 - prediction_smoothing) * smoothed_prediction
        
        # Compute velocity command from admittance
        vel_cmd = np.array([
            admittance_gains[0] * smoothed_prediction[0],
            admittance_gains[1] * smoothed_prediction[1],
            0.0
        ])
        
        # Get action from policy
        action, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, action)
        
        rollout.append(state.pipeline_state)
        actual_forces.append(np.asarray(state.info["force_current_vector"]))
        estimated_forces.append(smoothed_prediction.copy())
        velocity_commands.append(vel_cmd.copy())
        
        if step % 100 == 0:
            actual_mag = np.linalg.norm(actual_forces[-1])
            est_mag = np.linalg.norm(estimated_forces[-1])
            print(f"  Step {step}/{args.num_steps} | Actual: {actual_mag:.2f}N | Est: {est_mag:.2f}N")
    
    actual_forces = np.asarray(actual_forces)
    estimated_forces = np.asarray(estimated_forces)
    velocity_commands = np.asarray(velocity_commands)
    
    if args.use_zero:
        mode_label = "ZERO (ablation)"
        est_label = "Zero"
    elif args.use_noise:
        mode_label = "NOISE (ablation)"
        est_label = "Noise"
    else:
        mode_label = "Estimator"
        est_label = "Estimated"
    
    print(f"\nForce stats ({mode_label}):")
    print(f"  Actual force mag: mean={np.linalg.norm(actual_forces, axis=1).mean():.2f}, max={np.linalg.norm(actual_forces, axis=1).max():.2f}")
    print(f"  {est_label} force mag: mean={np.linalg.norm(estimated_forces, axis=1).mean():.2f}, max={np.linalg.norm(estimated_forces, axis=1).max():.2f}")
    
    # Compute direction error (cosine similarity)
    actual_norms = np.linalg.norm(actual_forces, axis=1, keepdims=True) + 1e-6
    est_norms = np.linalg.norm(estimated_forces, axis=1, keepdims=True) + 1e-6
    actual_unit = actual_forces / actual_norms
    est_unit = estimated_forces / est_norms
    cosine_sim = np.sum(actual_unit * est_unit, axis=1)
    valid_mask = actual_norms.squeeze() > 0.5  # Only when force is significant
    if np.any(valid_mask):
        print(f"  Direction accuracy (cosine sim): mean={cosine_sim[valid_mask].mean():.3f}")
    
    print("\nRendering video...")
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Render frames
    frames = []
    model = env.sys.mj_model
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Get torso body ID
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "base_link")
    
    try:
        for idx, (pipeline_state, actual, estimated, vel_cmd) in enumerate(
            zip(rollout[:-1], actual_forces, estimated_forces, velocity_commands)
        ):
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            data.qpos[:] = np.asarray(pipeline_state.q)
            data.qvel[:] = np.asarray(pipeline_state.qd)
            data.xfrc_applied[:] = np.asarray(pipeline_state.xfrc_applied)
            mujoco.mj_forward(model, data)
            
            # Use custom camera for better view
            torso_pos = data.xpos[torso_body_id]
            
            # Create a camera that follows the robot from behind and above
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = torso_body_id
            cam.distance = 1.5  # Zoom out more
            cam.azimuth = 135   # Behind and to the side
            cam.elevation = -25  # Looking down slightly
            cam.lookat[:] = torso_pos
            
            renderer.update_scene(data, camera=cam)
            
            arrow_origin = torso_pos + np.array([0.0, 0.0, 0.35])
            
            # Red arrow: Actual force
            if np.linalg.norm(actual) > 0.1:
                draw_force_arrow(
                    renderer, arrow_origin, actual,
                    scale=0.03, rgba=(1.0, 0.0, 0.0, 0.9)
                )
            
            # Blue arrow: Estimated force (slightly offset)
            if np.linalg.norm(estimated) > 0.1:
                draw_force_arrow(
                    renderer, arrow_origin + np.array([0.0, 0.0, 0.05]), estimated,
                    scale=0.03, rgba=(0.0, 0.0, 1.0, 0.9)
                )
            
            # Green arrow: Velocity command (smaller, different origin)
            if np.linalg.norm(vel_cmd) > 0.01:
                draw_force_arrow(
                    renderer, arrow_origin + np.array([0.0, 0.0, -0.1]), vel_cmd * 3,
                    scale=0.1, rgba=(0.0, 1.0, 0.0, 0.7)
                )
            
            frame = renderer.render()
            frames.append(frame)
            
            if idx % 100 == 0:
                print(f"  Rendered {idx}/{len(rollout)-1} frames")
        
        # Save video
        media.write_video(str(output_path), frames, fps=50)
        print(f"\nVideo saved to: {output_path}")
        
        # Also save a plot of force comparison
        plot_path = output_path.with_suffix('.png')
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        time = np.arange(len(actual_forces)) * 0.02  # 20ms per step
        
        for i, (ax, label) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
            ax.plot(time, actual_forces[:, i], 'r-', label=f'Actual {label}', alpha=0.7)
            ax.plot(time, estimated_forces[:, i], 'b--', label=f'Estimated {label}', alpha=0.7)
            ax.set_ylabel(f'Force {label} (N)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        if args.use_zero:
            title = 'ABLATION: Ground Truth vs Zero Input'
        elif args.use_noise:
            title = 'ABLATION: Ground Truth vs Random Noise'
        else:
            title = 'Force Estimation: Ground Truth vs Predicted'
        axes[0].set_title(title)
        
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150)
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Rendering failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        renderer.close()


if __name__ == "__main__":
    main()

