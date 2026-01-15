# Section 0: Standard library imports
import sys
from pathlib import Path

import numpy as np

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.experiment import run_experiment


# Section 1: Generalization configuration
# Agent counts to test (generalization range: 5 ± 2)
agent_counts = [6, 7]  # One-off: only running 6 and 7 (3, 4, 5 already completed)

# Override base_config from experiment.py (optional)
# Leave as None to use experiment.py defaults, or override specific values
config_override = None  # Example: {"env_name": "commons_harvest__closed"}


# Section 2: Generalization execution
def run_generalization():
    """Run experiments across different agent counts."""
    print(f"\n{'='*60}")
    print(f"STARTING GENERALIZATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Testing across {len(agent_counts)} agent counts: {agent_counts}")
    print(f"{'='*60}\n")
    
    all_experiments = {}
    
    for agent_count in agent_counts:
        print(f"\nRunning with {agent_count} agents...")
        
        # Create config override for this agent count
        run_config = {} if config_override is None else config_override.copy()
        run_config["num_players"] = agent_count
        
        # Run experiment
        results, exp_name = run_experiment(config_override=run_config)
        all_experiments[agent_count] = {
            "results": results,
            "experiment_name": exp_name
        }
        
        print(f"✓ Completed {agent_count} agents experiment\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"GENERALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nAll experiments completed:")
    for agent_count in agent_counts:
        exp_name = all_experiments[agent_count]["experiment_name"]
        final_welfares = [r["cumulative_welfare"][-1] for r in all_experiments[agent_count]["results"]]
        mean_welfare = np.mean(final_welfares)
        std_welfare = np.std(final_welfares)
        print(f"  {agent_count} agents: {mean_welfare:.2f} ± {std_welfare:.2f} → {exp_name}")
    print(f"\n{'='*60}\n")
    
    return all_experiments


if __name__ == "__main__":
    experiments = run_generalization()
    print("Generalization experiment complete!")
    print("\nNext steps:")
    print("  1. Run dataprocess.py to visualize individual experiments")
    print("  2. Compare performance across agent counts")

