#!/usr/bin/env python3
"""Verify force visualization with a trained policy executing velocity commands."""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import jax
from jax import numpy as jp
import mediapy as media
import numpy as np

from pupperv3_mjx import config, environment, utils


def load_policy_from_json(policy_path: Path):
    """Load policy network from exported JSON file."""
    with open(policy_path) as f:
        policy_dict = json.load(f)

    layers = policy_dict["layers"]
    weights = []
    for layer in layers:
        kernel = jp.array(layer["weights"][0])
        bias = jp.array(layer["weights"][1])
        weights.append((kernel, bias))

    activation_fn = utils.activation_fn_map(layers[0]["activation"])

    def policy_fn(obs, rng):
        x = obs
        for i, (kernel, bias) in enumerate(weights):
            x = x @ kernel + bias
            if i < len(weights) - 1:
                x = activation_fn(x)
            else:
                # Final layer uses tanh
                x = jp.tanh(x)
        return x, {}

    return policy_fn, policy_dict


def main() -> None:
    project_root = Path(__file__).resolve().parent
    checkpoint_base = (
        project_root
        / "output_morning-jazz-49"
    )
    policy_json_path = checkpoint_base / "policy.json"

    if not policy_json_path.exists():
        raise FileNotFoundError(f"Policy JSON not found at {policy_json_path}")

    # Load policy
    policy_fn, policy_dict = load_policy_from_json(policy_json_path)
    jit_policy = jax.jit(policy_fn)

    # Setup environment matching policy config
    xml_path = (
        project_root.parent
        / "pupper_v3_description"
        / "description"
        / "mujoco_xml"
        / "pupper_v3_complete.mjx.position.xml"
    )

    reward_config = config.get_config()

    env = environment.PupperV3Env(
        path=str(xml_path),
        reward_config=reward_config,
        action_scale=policy_dict["action_scale"],
        observation_history=policy_dict["observation_history"],
        dof_damping=policy_dict["kd"],
        position_control_kp=policy_dict["kp"],
        joint_lower_limits=policy_dict["joint_lower_limits"],
        joint_upper_limits=policy_dict["joint_upper_limits"],
        default_pose=jp.array(policy_dict["default_joint_pos"]),
        use_imu=policy_dict["use_imu"],
        # Force settings for visualization
        force_probability=0.8,
        force_duration_range=jp.array([40, 120]),
        force_magnitude_range=jp.array([3, 5]),
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Command sequence: stand, forward, backward, left, right, turn left, turn right
    vx, vy, wz = 0.5, 0.4, 1.5
    command_seq = jp.array([
        [0.0, 0.0, 0.0],
        [vx, 0.0, 0.0],
        [-vx, 0.0, 0.0],
        [0.0, vy, 0.0],
        [0.0, -vy, 0.0],
        [0.0, 0.0, wz],
        [0.0, 0.0, -wz],
    ])

    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)
    state.info["command"] = command_seq[0]
    rollout = [state.pipeline_state]

    n_steps = 700  # ~14 seconds at 50Hz
    command_change_interval = 100  # Change command every 100 steps (2 sec)

    print(f"Running policy rollout with {n_steps} steps...")
    for i in range(n_steps):
        rng, act_rng = jax.random.split(rng)

        # Change command periodically
        cmd_idx = (i // command_change_interval) % len(command_seq)
        state.info["command"] = command_seq[cmd_idx]

        action, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)

    print(f"Collected {len(rollout)} frames, rendering...")
    frames = env.render(rollout, camera="tracking_cam", force_vis_scale=0.08)

    output_dir = project_root / "videos"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "force_visualization_with_policy.mp4"

    fps = int(np.round(1.0 / env.dt))
    media.write_video(str(output_path), frames, fps=fps)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

