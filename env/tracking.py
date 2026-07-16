# Library imports
import mujoco

# Local imports
from config import TRACKING
from env import observations

class GripperTrackers:
    """Update methods for conditional checks and counters."""

    def __init__(self): pass

    def update(self, model, data, info):
        """Run all tracker checks at once."""

        self._update_height_checks(data, info)
        self._update_crush_checks(model, data, info)
        self._update_grasp_checks(model, data, info)
        self._update_steps(info)
    
    def _update_height_checks(self, data, info):
        """Check object-lifted status after initial rendering, allowing drop checks."""

        # Initial lift tracking (sticky)
        object_z = data.xpos[info["object_body_id"]][2]
        if object_z >= TRACKING["lifted_height"] or info["was_lifted"]:
            info["was_lifted"] = True

        # Success height step-counter
        object_z = data.xpos[info["object_body_id"]][2]
        if object_z >= TRACKING["success_height"]:
            info["hold_counter"] += 1
        else:
            info["hold_counter"] = 0
        
        # Success update via hold_counter check
        if info["hold_counter"] >= TRACKING["hold_steps_required"]:
            info["success"] = True
    
    def _update_crush_checks(self, model, data, info):
        """Track crushing and crushed states, including increment step counter."""

        # Crush tracker
        if observations.is_crushing(model, data, info):
            info["object_crushing"] = True
        else:
            info["object_crushing"] = False
        
        # Crush step-counter
        if info["object_crushing"]:
            info["object_crushing_counter"] += 1
        else:
            info["object_crushing_counter"] = 0

        # Fully-crushed check
        if info["object_crushing_counter"] >= TRACKING["crush_steps_required"]:
            info["object_crushed"] = True

    def _update_grasp_checks(self, model, data, info):
        """Sticky did_grasp gate, plus just_grasped — True only on the exact step of first grasp."""
    
        currently_grasping = observations.is_grasping(model, data, info)
        info["just_grasped"] = currently_grasping and not info["did_grasp"]
    
        if currently_grasping:
            info["did_grasp"] = True

    def _update_steps(self, info):
        """Increment total steps taken this episode."""
        info["steps_complete"] += 1

class GripperDoneConditions:
    """Update methods for done conditions."""
    def __init__(self): pass

    def done_conditions(self, info):
        """Run full check for done conditions."""
        return info["success"] or (info["steps_complete"] >= TRACKING["maximum_steps"]) or info["object_crushed"]