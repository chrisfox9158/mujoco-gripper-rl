# Library imports
import mujoco
import numpy as np

# Local imports
from config import RANDOMIZATION
from config import TRACKING
from env import observations
from env import rewards

class GripperEnv:
    def __init__(self, xml_path, obs_extractors, reward_terms):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.obs_extractors = obs_extractors
        self.reward_terms = reward_terms
        self.action_dim = self.model.nu

    def reset(self):
        """Reset the simulation back to default, with randomized object values."""

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        mass_range = RANDOMIZATION["object_mass"]
        threshold_range = RANDOMIZATION["crush_threshold"]

        random_mass_value = np.random.uniform(mass_range["min"], mass_range["max"])
        random_threshold_value = np.random.uniform(threshold_range["min"], threshold_range["max"])

        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self.model.body_mass[self.object_id] = random_mass_value

        self.info = {"crush_threshold": random_threshold_value, "object_body_id": self.object_id, "was_lifted": False}

        obs = self._build_obs()
        self.obs_dim = len(obs)
        return obs
    
    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        object_z = self.data.xpos[self.info["object_body_id"]][2]
        if object_z >= TRACKING["lifted_height"] or self.info["was_lifted"]:
            self.info["was_lifted"] = True

        obs = self._build_obs()
        reward = self._compute_reward()

        return obs, reward
    
    def _compute_reward(self):
        return sum(term(self.model, self.data, self.info) for term in self.reward_terms)

    def _build_obs(self):
        return np.concatenate([f(self.model, self.data) for f in self.obs_extractors])
    
# Output testing
if __name__ == "__main__":
    env = GripperEnv(xml_path="three_finger_two_joint_gripper.xml", obs_extractors=[observations.obs_joint_angles], reward_terms=[rewards.reward_drop_penalty])
    obs = env.reset()
    action = np.zeros(env.action_dim)
    obs, reward = env.step(action)
    print(obs, obs.shape, reward)