# Library imports
import mujoco
import numpy as np

# Local imports


# Observation handlers
def obs_joint_angles(model, data):
    """Observes joint angles and returns as a NumPy array."""

    joint_names = ["palm_lift", "finger1_joint1", "finger1_joint2", "finger2_joint1", "finger2_joint2", "finger3_joint1", "finger3_joint2"]
    angles = []
    
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = model.jnt_qposadr[joint_id]
        angles.append(data.qpos[qpos_adr])

    return np.array(angles)

def obs_touch_sensors(model, data):
    """Observes touch sensor data and returns as a NumPy array."""

    sensor_names = ["finger1_touch", "finger2_touch", "finger3_touch"]
    readings = []
    
    for name in sensor_names:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = model.sensor_adr[sensor_id]
        dim = model.sensor_dim[sensor_id]
        value = data.sensordata[adr : adr + dim]
        readings.append(value)

    return np.concatenate(readings)

def is_crushing(model, data, info):
    """Checks if the gripper is crushing the object."""
    
    touch_readings = obs_touch_sensors(model, data)
    return np.any(touch_readings > info["crush_threshold"])