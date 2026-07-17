# Library imports
import json
import os

# Exporter
def export_run(logger, run_dir):
    """Write the logger's accumulated episode data to disk as JSON."""
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(logger.episodes, f, indent=2)