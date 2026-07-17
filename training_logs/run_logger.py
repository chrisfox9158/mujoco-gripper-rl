class RunLogger:
    def __init__(self):
        self.episodes = []

    def log_episode(self, episode_num, total_reward, term_totals, steps, info, noise_scale):
        self.episodes.append({
            "episode": episode_num,
            "total_reward": total_reward,
            "term_totals": term_totals,
            "steps": steps,
            "outcome": self._get_outcome(info),
            "noise_scale": noise_scale,
        })

    def _get_outcome(self, info):
        if info["success"]:
            return "success"
        elif info["object_crushed"]:
            return "crushed"
        else:
            return "timeout"