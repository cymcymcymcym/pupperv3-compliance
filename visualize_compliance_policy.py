#!/usr/bin/env python3
"""
Visualize the compliance policy trained with force estimator.

Shows:
- The robot walking with external force perturbations
- Red arrow: Actual applied force
- Blue arrow: Estimated force (from force estimator)
- Green arrow: Velocity command derived from admittance

Usage:
    python visualize_compliance_policy.py \
        --policy-dir /path/to/output_compliance_2025-11-30_11-16-11 \
        --force-estimator-path /path/to/force_estimator.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from jax import numpy as jp
import mediapy as media
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from pupperv3_mjx import config, environment, utils


def load_policy_from_json(policy_path: Path):
    """Load policy from JSON file."""
    with open(policy_path) as f:
        policy_dict = json.load(f)

    layers = policy_dict["layers"]
    weights = []
    for layer in layers:
        kernel = jp.array(layer["weights"][0])
        bias = jp.array(layer["weights"][1])
        weights.append((kernel, bias))

    activation_fn = utils.activation_fn_map(layers[0]["activation"])

    def policy_fn(obs: jp.ndarray, rng: jp.ndarray) -> Tuple[jp.ndarray, dict]:
        x = obs
        for i, (kernel, bias) in enumerate(weights):
            x = x @ kernel + bias
            if i < len(weights) - 1:
                x = activation_fn(x)
            else:
                x = jp.tanh(x)
        return x, {}

    return policy_fn, policy_dict


def load_force_estimator(estimator_path: Path):
    """Load force estimator from JSON file."""
    with open(estimator_path) as f:
        est_dict = json.load(f)

    layers = est_dict["layers"]
    input_mean = jp.array(est_dict["input_mean"])
    input_std = jp.array(est_dict["input_std"])

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
        x = (obs - input_mean) / (input_std + 1e-6)
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

    return estimator_fn


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
    parser = argparse.ArgumentParser(description="Visualize compliance policy")
    parser.add_argument(
        "--policy-dir",
        type=str,
        required=True,
        help="Path to policy output directory (contains policy.json)"
    )
    parser.add_argument(
        "--force-estimator-path",
        type=str,
        required=True,
        help="Path to force_estimator.json"
    )
    parser.add_argument(
        "--admittance-gains",
        type=str,
        default="0.25,0.25",
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
        default="videos/compliance_demo.mp4",
        help="Output video path"
    )
    
    args = parser.parse_args()
    
    admittance_gains = tuple(float(x) for x in args.admittance_gains.split(","))
    
    # Find policy.json in the directory
    policy_dir = Path(args.policy_dir)
    policy_path = policy_dir / "policy.json"
    
    if not policy_path.exists():
        # Try to find it
        possible_paths = list(policy_dir.glob("**/policy.json"))
        if possible_paths:
            policy_path = possible_paths[0]
        else:
            raise FileNotFoundError(f"policy.json not found in {policy_dir}")
    
    print(f"Loading policy from: {policy_path}")
    print(f"Loading force estimator from: {args.force_estimator_path}")
    print(f"Admittance gains: {admittance_gains}")
    
    # Load policy and force estimator
    policy_fn, policy_dict = load_policy_from_json(policy_path)
    estimator_fn = load_force_estimator(Path(args.force_estimator_path))
    
    jit_policy = jax.jit(policy_fn)
    jit_estimator = jax.jit(estimator_fn)
    
    # Setup environment
    reward_config = config.get_config()
    
    # Set reward scales for compliance
    reward_config.rewards.scales.tracking_lin_vel = 1.5
    reward_config.rewards.scales.tracking_ang_vel = 0.0
    reward_config.rewards.scales.tracking_orientation = 1.0
    reward_config.rewards.scales.lin_vel_z = -2.0
    reward_config.rewards.scales.ang_vel_xy = -0.05
    reward_config.rewards.scales.orientation = -5.0
    reward_config.rewards.scales.torques = -0.0002
    reward_config.rewards.scales.joint_acceleration = -1e-6
    reward_config.rewards.scales.action_rate = -0.01
    reward_config.rewards.scales.feet_air_time = 0.05
    reward_config.rewards.scales.stand_still = -0.5
    reward_config.rewards.scales.stand_still_joint_velocity = -0.1
    reward_config.rewards.scales.abduction_angle = -0.1
    reward_config.rewards.scales.termination = -100.0
    reward_config.rewards.scales.foot_slip = -0.1
    reward_config.rewards.scales.knee_collision = -1.0
    reward_config.rewards.scales.body_collision = -1.0
    reward_config.rewards.scales.force_following = 0.0
    reward_config.rewards.tracking_sigma = 0.25
    
    env = environment.PupperV3Env(
        path=args.model_path,
        reward_config=reward_config,
        action_scale=policy_dict["action_scale"],
        observation_history=policy_dict["observation_history"],
        dof_damping=policy_dict["kd"],
        position_control_kp=policy_dict["kp"],
        joint_lower_limits=policy_dict["joint_lower_limits"],
        joint_upper_limits=policy_dict["joint_upper_limits"],
        default_pose=jp.array(policy_dict["default_joint_pos"]),
        use_imu=policy_dict["use_imu"],
        force_probability=0.8,
        force_duration_range=jp.array([40, 120]),
        force_magnitude_range=jp.array([3, 6]),
    )
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # Run simulation
    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)
    
    rollout = [state.pipeline_state]
    actual_forces = []
    estimated_forces = []
    velocity_commands = []
    
    # EMA smoothing for predictions
    smoothed_prediction = np.zeros(3)
    prediction_smoothing = 0.2
    
    print(f"Running {args.num_steps} steps...")
    
    for step in range(args.num_steps):
        rng, act_rng = jax.random.split(rng)
        
        # Get force estimate
        raw_prediction = np.asarray(jit_estimator(state.obs))
        smoothed_prediction = prediction_smoothing * raw_prediction + (1 - prediction_smoothing) * smoothed_prediction
        
        # Compute velocity command from admittance
        vel_cmd = np.array([
            admittance_gains[0] * smoothed_prediction[0],
            admittance_gains[1] * smoothed_prediction[1],
            0.0
        ])
        
        # For visualization, we use the original environment (not the estimator one)
        # So the command is whatever was sampled by the environment
        action, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, action)
        
        rollout.append(state.pipeline_state)
        actual_forces.append(np.asarray(state.info["force_current_vector"]))
        estimated_forces.append(smoothed_prediction.copy())
        velocity_commands.append(vel_cmd.copy())
        
        if step % 100 == 0:
            print(f"  Step {step}/{args.num_steps}")
    
    actual_forces = np.asarray(actual_forces)
    estimated_forces = np.asarray(estimated_forces)
    velocity_commands = np.asarray(velocity_commands)
    
    print("Rendering video...")
    
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
            
            renderer.update_scene(data, camera="tracking_cam")
            
            torso_pos = data.xpos[torso_body_id]
            arrow_origin = torso_pos + np.array([0.0, 0.0, 0.35])
            
            # Red arrow: Actual force
            if np.linalg.norm(actual) > 0.1:
                draw_force_arrow(
                    renderer,
                    arrow_origin,
                    actual,
                    scale=0.08,
                    rgba=(1.0, 0.0, 0.0, 1.0),
                )
            
            # Blue arrow: Estimated force
            if np.linalg.norm(estimated) > 0.1:
                draw_force_arrow(
                    renderer,
                    arrow_origin + np.array([0.0, 0.0, 0.05]),
                    estimated,
                    scale=0.08,
                    rgba=(0.0, 0.3, 1.0, 1.0),
                )
            
            # Green arrow: Velocity command (scaled up for visibility)
            vel_3d = np.array([vel_cmd[0], vel_cmd[1], 0.0])
            if np.linalg.norm(vel_3d) > 0.01:
                draw_force_arrow(
                    renderer,
                    arrow_origin + np.array([0.0, 0.0, 0.10]),
                    vel_3d * 2.0,  # Scale up velocity for visibility
                    scale=0.15,
                    rgba=(0.0, 1.0, 0.0, 1.0),
                )
            
            frame = renderer.render()
            frames.append(frame)
    finally:
        renderer.close()
    
    # Save video
    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fps = int(np.round(1.0 / env.dt))
    media.write_video(str(output_path), frames, fps=fps)
    print(f"Wrote video to {output_path}")
    
    # Plot force magnitudes
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Force magnitude plot
    ax1 = axes[0]
    actual_mag = np.linalg.norm(actual_forces, axis=1)
    estimated_mag = np.linalg.norm(estimated_forces, axis=1)
    timesteps = np.arange(len(actual_mag))
    ax1.plot(timesteps, actual_mag, label="Actual |F|", color="red", alpha=0.8)
    ax1.plot(timesteps, estimated_mag, label="Estimated |F|", color="blue", alpha=0.8)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Force magnitude (N)")
    ax1.set_title("External Force: Actual vs Estimated")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Velocity command plot
    ax2 = axes[1]
    ax2.plot(timesteps, velocity_commands[:, 0], label="Vel X cmd", color="green", alpha=0.8)
    ax2.plot(timesteps, velocity_commands[:, 1], label="Vel Y cmd", color="orange", alpha=0.8)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Velocity command (m/s)")
    ax2.set_title("Velocity Commands from Admittance Controller")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_path = output_path.parent / "compliance_analysis.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Wrote plot to {plot_path}")
    
    # Print summary stats
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Actual force magnitude: mean={actual_mag.mean():.3f}, max={actual_mag.max():.3f} N")
    print(f"Estimated force magnitude: mean={estimated_mag.mean():.3f}, max={estimated_mag.max():.3f} N")
    
    # Compute direction error (only when force is significant)
    mask = actual_mag > 0.5
    if mask.sum() > 0:
        actual_unit = actual_forces[mask] / (actual_mag[mask, None] + 1e-6)
        estimated_unit = estimated_forces[mask] / (estimated_mag[mask, None] + 1e-6)
        cos_sim = np.sum(actual_unit * estimated_unit, axis=1)
        angle_error = np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi
        print(f"Direction error (when |F|>0.5N): mean={angle_error.mean():.1f}°, median={np.median(angle_error):.1f}°")
    
    print("=" * 60)


if __name__ == "__main__":
    main()


