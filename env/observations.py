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