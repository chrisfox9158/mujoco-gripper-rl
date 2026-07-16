# Library imports
import mujoco
import numpy as np

# Local imports
from config import HARDWARE

# Observation handlers
def obs_joint_angles(model, data):
    """Observes joint angles and returns as a NumPy array."""

    joint_names = HARDWARE["joint_names"]
    angles = []
    
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = model.jnt_qposadr[joint_id]
        angles.append(data.qpos[qpos_adr])

    return np.array(angles)

def obs_touch_sensors(model, data):
    """Observes touch sensor data and returns as a NumPy array."""

    sensor_names = HARDWARE["sensor_names"]
    readings = []
    
    for name in sensor_names:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = model.sensor_adr[sensor_id]
        dim = model.sensor_dim[sensor_id]
        value = data.sensordata[adr : adr + dim]
        readings.append(value)

    return np.concatenate(readings)

def obs_joint_velocities(model, data):
    """Observes joint angular velocities and returns as a NumPy array."""

    joint_names = HARDWARE["joint_names"]
    velocities = []

    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qvel_adr = model.jnt_dofadr[joint_id]
        velocities.append(data.qvel[qvel_adr])
    
    return np.array(velocities)

def get_fingertip_positions(model, data):
    """Returns fingertip world positions as a (3, 3) array."""

    site_names = HARDWARE["site_names"]
    positions = []

    for name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        positions.append(data.site_xpos[site_id])

    return np.array(positions)

def get_average_distance(model, data, info):
    """Returns average straight-line distance to object across all fingertips."""

    fingertip_positions = get_fingertip_positions(model, data)
    object_position = data.xpos[info["object_body_id"]]

    distances = np.linalg.norm(fingertip_positions - object_position, axis=1)
    return np.mean(distances)

def is_crushing(model, data, info):
    """Checks if the gripper is crushing the object."""
    
    touch_readings = obs_touch_sensors(model, data)
    return np.any(touch_readings > info["crush_threshold"])

def is_grasping(model, data, info):
    """Checks if the gripper is grasping the object."""

    touch_readings = obs_touch_sensors(model, data)
    return np.all(touch_readings >= info["grasp_threshold"])