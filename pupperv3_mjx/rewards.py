import jax
from jax import numpy as jp
from brax.base import Motion, Transform
from brax import base, math
import numpy as np

EPS = 1e-6
# ------------ reward functions----------------
def reward_lin_vel_z(xd: Motion) -> jax.Array:
    # Penalize z axis base linear velocity
    return jp.clip(jp.square(xd.vel[0, 2]), -1000.0, 1000.0)


def reward_ang_vel_xy(xd: Motion) -> jax.Array:
    # Penalize xy axes base angular velocity
    return jp.clip(jp.sum(jp.square(xd.ang[0, :2])), -1000.0, 1000.0)


def reward_tracking_orientation(
    desired_world_z_in_body_frame: jax.Array, x: Transform, tracking_sigma: float
) -> jax.Array:
    # Tracking of desired body orientation
    world_z = jp.array([0.0, 0.0, 1.0])
    world_z_in_body_frame = math.rotate(world_z, math.quat_inv(x.rot[0]))
    error = jp.sum(jp.square(world_z_in_body_frame - desired_world_z_in_body_frame))
    return jp.clip(jp.exp(-error / (tracking_sigma + EPS)), -1000.0, 1000.0)


def reward_orientation(x: Transform) -> jax.Array:
    # Penalize non flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.clip(jp.sum(jp.square(rot_up[:2])), -1000.0, 1000.0)


def reward_torques(torques: jax.Array) -> jax.Array:
    # Penalize torques
    # This has a sparifying effect
    # return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))
    # Use regular sum-squares like in LeggedGym
    return jp.clip(jp.sum(jp.square(torques)), -1000.0, 1000.0)


def reward_joint_acceleration(
    joint_vel: jax.Array, last_joint_vel: jax.Array, dt: float
) -> jax.Array:
    return jp.clip(jp.sum(jp.square((joint_vel - last_joint_vel) / (dt + EPS))), -1000.0, 1000.0)


def reward_mechanical_work(torques: jax.Array, velocities: jax.Array) -> jax.Array:
    # Penalize mechanical work
    return jp.clip(jp.sum(jp.abs(torques * velocities)), -1000.0, 1000.0)


def reward_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    # Penalize changes in actions
    return jp.clip(jp.sum(jp.square(act - last_act)), -1000.0, 1000.0)


def reward_tracking_lin_vel(
    commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
) -> jax.Array:
    # Tracking of linear velocity commands (xy axes)
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(-lin_vel_error / (tracking_sigma + EPS))
    return jp.clip(lin_vel_reward, -1000.0, 1000.0)


def reward_tracking_ang_vel(
    commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
) -> jax.Array:
    # Tracking of angular velocity commands (yaw)
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.clip(jp.exp(-ang_vel_error / (tracking_sigma + EPS)), -1000.0, 1000.0)


def reward_feet_air_time(
    air_time: jax.Array,
    first_contact: jax.Array,
    commands: jax.Array,
    minimum_airtime: float = 0.1,
) -> jax.Array:
    # Reward air time.
    rew_air_time = jp.sum((air_time - minimum_airtime) * first_contact)
    rew_air_time *= math.normalize(commands[:3])[1] > 0.05  # no reward for zero command
    return jp.clip(rew_air_time, -1000.0, 1000.0)


def reward_abduction_angle(
    joint_angles: jax.Array, desired_abduction_angles: jax.Array = jp.zeros(4)
):
    # Penalize abduction angle
    return jp.clip(jp.sum(jp.square(joint_angles[1::3] - desired_abduction_angles)), -1000.0, 1000.0)


def reward_stand_still(
    commands: jax.Array,
    joint_angles: jax.Array,
    default_pose: jax.Array,
    command_threshold: float,
) -> jax.Array:
    """
    Penalize motion at zero commands
    Args:
        commands: robot velocity commands
        joint_angles: joint angles
        default_pose: default pose
        command_threshold: if norm of commands is less than this, return non-zero penalty
    """

    # Penalize motion at zero commands
    return jp.clip(
        jp.sum(jp.abs(joint_angles - default_pose)) * (
            math.normalize(commands[:3])[1] < command_threshold
        ),
        -1000.0,
        1000.0
    )


def reward_foot_slip(
    pipeline_state: base.State,
    contact_filt: jax.Array,
    feet_site_id: np.array,
    lower_leg_body_id: np.array,
) -> jax.Array:
    # get velocities at feet which are offset from lower legs
    # pytype: disable=attribute-error
    pos = pipeline_state.site_xpos[feet_site_id]  # feet position
    feet_offset = pos - pipeline_state.xpos[lower_leg_body_id]
    # pytype: enable=attribute-error
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = lower_leg_body_id - 1  # we got rid of the world body
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel
    # Penalize large feet velocity for feet that are in contact with the ground.
    return jp.clip(
        jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1))),
        -1000.0,
        1000.0
    )


def reward_termination(done: jax.Array, step: jax.Array, step_threshold: int) -> jax.Array:
    return done & (step < step_threshold)


def reward_geom_collision(pipeline_state: base.State, geom_ids: np.array) -> jax.Array:
    contact = jp.array(0.0)
    for id in geom_ids:
        contact += jp.sum(
            ((pipeline_state.contact.geom1 == id) | (pipeline_state.contact.geom2 == id))
            * (pipeline_state.contact.dist < 0.0)
        )
    return jp.clip(contact, -1000.0, 1000.0)

def reward_force_following(
    pipeline_state: base.State, 
    torso_body_idx: int,
    tracking_sigma: float = 0.5
) -> jax.Array:
    """
    Reward for moving compliantly in response to applied force.
    Encourages robot to "go with the flow" of external forces with appropriate magnitude.
    
    This reward combines:
    1. Direction alignment (move in force direction)
    2. Compliance (accelerate appropriately with force magnitude)
    
    Args:
        pipeline_state: The current pipeline state
        torso_body_idx: Index of the torso/base body (usually 0)
        tracking_sigma: Scales the reward sensitivity
    
    Returns:
        Reward value (higher when velocity aligns with force and responds appropriately)
    """
    # Extract force (first 3 elements of wrench - xfrc_applied format is [force, torque])
    force = pipeline_state.xfrc_applied[torso_body_idx, :3]  # [fx, fy, fz]
    velocity = pipeline_state.xd.vel[torso_body_idx]         # [vx, vy, vz]
    acceleration = pipeline_state.xd.ang[torso_body_idx]     # Using as proxy for body acceleration
    
    # Compute force magnitude
    force_mag = jp.linalg.norm(force)
    
    # Only compute reward when force is active
    force_active = force_mag > 0.1  # Threshold for "force is active"
    
    # Component 1: Direction alignment
    force_direction = force / (force_mag + EPS)
    velocity_mag = jp.linalg.norm(velocity) + EPS
    velocity_direction = velocity / velocity_mag
    
    # Alignment: 1 when parallel, -1 when opposite
    alignment = jp.dot(force_direction, velocity_direction)
    directional_error = 1.0 - alignment  # 0 when aligned, 2 when opposite
    
    # Component 2: Magnitude response (compliance)
    # Idea: Power = F · v. If robot is compliant, it should move proportionally to force
    # High power with low velocity = resisting (bad)
    # Low power with high velocity = already moving freely (neutral)
    power = jp.dot(force, velocity)  # Can be negative if opposing
    
    # Normalize power by force magnitude to get "compliance velocity"
    # If moving with force: power > 0, good
    # If resisting force: power < 0, bad
    compliance = power / (force_mag + EPS)
    
    # Penalize resistance (negative compliance) and reward yielding (positive compliance)
    # Target: compliance should be proportional to force magnitude
    # For a 10N force, we might expect ~0.5 m/s compliance velocity
    target_compliance = force_mag * 0.05  # 0.05 m/s per Newton (tunable parameter)
    compliance_error = jp.square(compliance - target_compliance)
    
    # Combined error: both direction and magnitude matter
    #total_error = directional_error + compliance_error
    total_error = directional_error
    
    # Exponential reward
    reward = jp.exp(-total_error / (tracking_sigma + EPS))
    
    # Only give reward when force is active
    reward = jp.where(force_active, reward, 0.0)
    
    return jp.clip(reward, -1000.0, 1000.0)
