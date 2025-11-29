#!/usr/bin/env python3
"""Verify force arrow visualization by rendering a test rollout."""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from jax import numpy as jp
import mediapy as media
import numpy as np

from pupperv3_mjx import config, environment


def main() -> None:
    project_root = Path(__file__).resolve().parent
    xml_path = project_root / "test" / "test_pupper_model.xml"
    if not xml_path.exists():
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
        action_scale=0.5,
        observation_history=15,
        force_probability=0.9,
        force_duration_range=jp.array([30, 100]),
        force_magnitude_range=jp.array([3, 6]),
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    n_steps = 1000
    for _ in range(n_steps):
        rng, _ = jax.random.split(rng)
        action = jp.zeros(env.sys.nu)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)

    print(f"Collected {len(rollout)} frames, rendering...")
    frames = env.render(rollout, camera="tracking_cam", force_vis_scale=0.05)

    output_dir = project_root / "videos"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "force_visualization_test.mp4"

    fps = int(np.round(1.0 / env.dt))
    media.write_video(str(output_path), frames, fps=fps)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
