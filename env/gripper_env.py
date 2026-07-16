# Library imports
import mujoco
import numpy as np

# Local imports
from config import RANDOMIZATION
from config import TRACKING
from env import observations
from env import rewards
from env import tracking

class GripperEnv:
    def __init__(self, xml_path, obs_extractors, reward_terms):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.obs_extractors = obs_extractors
        self.reward_terms = reward_terms
        self.action_dim = self.model.nu
        self.trackers = tracking.GripperTrackers()
        self.done_conditions = tracking.GripperDoneConditions()

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

        self.info = {
            "crush_threshold": random_threshold_value,
            "object_body_id": self.object_id,
            "was_lifted": False,
            "hold_counter": 0,
            "success": False,
            "done": False,
            "steps_complete": 0,
            "object_crushing": False,
            "object_crushed": False,
            "object_crushing_counter": 0
            }

        obs = self._build_obs()
        self.obs_dim = len(obs)
        return obs
    
    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        self.trackers.update(self.model, self.data, self.info)
        self.info["done"] = self.done_conditions.done_conditions(self.info)

        obs = self._build_obs()
        reward = self._compute_reward()
        done = self.info["done"]

        return obs, reward, done
    
    def _compute_reward(self):
        return sum(term(self.model, self.data, self.info) for term in self.reward_terms)

    def _build_obs(self):
        return np.concatenate([f(self.model, self.data) for f in self.obs_extractors])
    
# Output testing
if __name__ == "__main__":
    env = GripperEnv(xml_path="three_finger_two_joint_gripper.xml", obs_extractors=[observations.obs_joint_angles, observations.obs_touch_sensors], reward_terms=[rewards.reward_drop_penalty, rewards.reward_crush_penalty, rewards.reward_success])
    obs = env.reset()
    action = np.zeros(env.action_dim)
    obs, reward, done = env.step(action)
    print(obs, obs.shape, reward, done)