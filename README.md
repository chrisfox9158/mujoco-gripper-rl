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
Environment (physics model, observation/reward pipeline, done-condition tracking) and agent (networks, replay buffer, full TD3 orchestration) are complete and tested. The TD3 implementation has been validated in isolation against a known benchmark task (see [Agent Validation](#agent-validation)). Real training on the MuJoCo gripper task is in progress. Early runs on local hardware surfaced and resolved several real bugs (reward exploits, MuJoCo solver instability, a runaway penalty bug) but were compute-constrained.

In order to continue testing, future training will be hosted on non-local hardware. The primary current option is Google Colab, which provides GPU access and longer, checkpointed runs via an attached Google Drive and snapshot system.

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
