### Plan for later agent, training, and run documentation filestructure ###
agent/
├── networks.py        # Actor, Critic — pure architecture
├── replay_buffer.py    # ReplayBuffer
└── td3.py              # TD3Agent — owns all 6 networks, update logic, save/load

logging/
├── run_logger.py        # collects per-episode stats DURING a run (in memory)
└── run_export.py         # writes everything to disk AT END of run (or checkpoint interval)

train.py                  # orchestrator: env + agent + logger tied together

runs/
└── <timestamp>/
    ├── checkpoints/
    │   ├── checkpoint_ep1000.pt
    │   └── checkpoint_ep2000.pt
    ├── summary.json         # full raw data dump — every episode's stats
    ├── overview.md           # ~20 entries, stats averaged every 5% of training
    └── plots/
        ├── reward_curve.png
        ├── success_rate.png
        └── term_breakdown.png