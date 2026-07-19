# MuJoCo Three-Finger Gripper — TD3 RL Implementation

## Purpose
A TD3-based reinforcement learning project involving a 3D-simulated, pressure-sensitive three-finger robotic gripper. The model is designed to learn to grasp and lift an object whose mass and crush thresholds are randomized per-episode, without dropping or crushing. The agent has no direct access to the randomized values and must learn to infer safe grip force from live touch-sensor feedback.

This project was designed to deepen my own understanding of the mechanics of continuous-control reinforcement learning. The MuJoCo physics model, reward and observation systems, and TD3 agent (networks, twin-critic system, delayed updates, target smoothing, and replay buffer) are implemented by hand in PyTorch and MuJoCo, without a pre-built RL framework.

## Design Highlights
- **Model-agnostic agent architecture** — The TD3 implementation (`agent/`) has no connection to the MuJoCo model hardware; `obs_dim`/`action_dim` are read from the loaded model at runtime, so the same agent code could train against a differently-shaped gripper without modification.
- **Modular reward and observation system** — Reward terms, observation extractors, and episode trackers are registered as independent functions or classes, such that additional task dimensions can be added without refactoring existing, working code.
- **Exploit-checked reward design** — Every current reward-shaping term (distance, lift-height) uses potential-based (delta) shaping to prevent reward-farming via oscillation or hovering; sticky/edge-triggered flags (`just_grasped`, `just_dropped`) prevent one-time events from firing continuously.
- **Independently validated agent implementation** — See [Agent Validation](#agent-validation) below.

## Status
Environment (physics model, observation/reward pipeline, done-condition tracking) and agent (networks, replay buffer, full TD3 orchestration) are complete and tested. The TD3 implementation has been validated in isolation against a known benchmark task (see [Agent Validation](#agent-validation)).

Real training on the MuJoCo task was attempted on local hardware. Early runs surfaced and resolved several minor bugs, including reward-shaping exploits, MuJoCo solver instability, and a runaway penalty system. However, reaching a meaningfully trained, near-complete policy requires a training budget (likely tens of thousands of episodes, based on early-run trends) beyond local hardware constraints or the practical compute available via free-tier cloud computing resources.

Rather than force an inadequate training run to completion, the training phase is deliberately concluded here. Design, implementation, and validation of the architecture are the primary outcomes of this project. A future attempt, given real compute access, would simply run `train.py` against the existing architecture; no further development work is required.

## Agent Validation
Before real training began, the TD3 implementation was validated in isolation against a known benchmark task, independent of this project's MuJoCo environment and reward design in a secondary repo: [td3-validation](https://github.com/chrisfox9158/td3-validation). Results confirm the six-network architecture, replay buffer, and update logic converge correctly, consistent with published TD3 benchmarks.

## References
Agentic structure implements TD3 strategy, matching the twin-critic and delayed-update structure from Fujimoto et al. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)

## Setup
```bash
uv venv
uv sync
```

## Usage
```bash
uv run python train.py
```

## Repository Structure
```
env/                  # GripperEnv, observation extractors, reward terms, episode trackers
agent/                # Actor/Critic networks, replay buffer, TD3 orchestration
training_logs/        # Run logger + export of per-run documentation into `runs/` files
config.py             # All tunable hyperparameters (randomization, tracking, rewards, agent, training)
train.py              # Training loop orchestrator
*.xml                 # MJCF model(s) for the gripper
runs/                 # Per-run output (checkpoints, logs, plots)
loader/               # Optional checkpoint loader, allowing continuation of training from checkpoint files
```
