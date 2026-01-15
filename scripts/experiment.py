# Section 0: Standard library imports
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.simulate import SimulationConfig, run_simulation


# Section 1: Experiment configuration - set your parameters here
# Base simulation configuration (will be passed to SimulationConfig)
base_config = {
    "env_name": "commons_harvest__open",
    "norm_type": "gpt5",  # Options: "gpt5", "claude", "static_apple_blocker", etc., or "None"
    "epsilon": 0.0,  # Norm compliance (0.0 = always obey, 1.0 = always ignore)
    "agent_type": "selfish",  # Options: "selfish" (more agent types coming soon)
    "num_players": 5,
    "timesteps": 1000,  # Standard experiment length is 1000 timesteps.
}

# Experiment settings
num_simulations = 10  # Number of independent runs

# Random seed settings (optional)
use_seeds = False  # Set to True if runs look too similar (adds reproducibility)
seed_start = 42  # Starting seed (will use 42, 43, 44... for each run)


# Section 2: Experiment execution
# Creates a descirptive folder name eg: commons_harvest_closed_gpt5_eps00_20251007_152030
def create_experiment_name():
    """Generate a descriptive name for this experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_str = base_config["norm_type"] if base_config["norm_type"] != "None" else "baseline"
    eps_str = f"eps{base_config['epsilon']}".replace(".", "")
    return f"{base_config['env_name']}_{norm_str}_{eps_str}_{timestamp}"


def save_experiment_data(experiment_name: str, all_results: list, config: dict = None):
    """
    Save experiment data to organized folder structure.
    
    Args:
        experiment_name: Unique name for this experiment
        all_results: List of result dictionaries from each simulation run
        config: Configuration dict to save (defaults to base_config)
    """
    if config is None:
        config = base_config
    # Create data directory structure
    data_dir = ROOT_DIR / "data" / experiment_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual run data
    for i, result in enumerate(all_results):
        run_file = data_dir / f"run_{i}.npy"
        np.save(run_file, result["social_welfare"])
    
    # Save cumulative welfare data
    cumulative_data = np.array([r["cumulative_welfare"] for r in all_results])
    cumulative_file = data_dir / "cumulative_welfare_all_runs.npy"
    np.save(cumulative_file, cumulative_data)
    
    # Save experiment configuration and metadata
    config_data = {
        "experiment_name": experiment_name,
        "config": {
            **config,  # Include all simulation config (base + overrides)
            "num_simulations": num_simulations,
            "use_seeds": use_seeds,
            "seed_start": seed_start if use_seeds else None,
        },
        "results_summary": {
            "num_runs": len(all_results),
            "mean_final_welfare": float(np.mean([r["cumulative_welfare"][-1] for r in all_results])),
            "std_final_welfare": float(np.std([r["cumulative_welfare"][-1] for r in all_results])),
        },
        "run_metadata": [r["metadata"] for r in all_results]
    }
    
    config_file = data_dir / "experiment_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Data saved to: {data_dir}")
    print(f"{'='*60}")
    print(f"  • Individual runs: run_0.npy ... run_{num_simulations-1}.npy")
    print(f"  • Cumulative data: cumulative_welfare_all_runs.npy")
    print(f"  • Configuration: experiment_config.json")
    print(f"{'='*60}\n")


def run_experiment(config_override: dict = None):
    """
    Main experiment loop - runs multiple simulations and saves results.
    
    Args:
        config_override: Optional dict to override base_config values
                         (e.g., {"num_players": 3})
    """
    # Merge config overrides
    exp_config = base_config.copy()
    if config_override:
        exp_config.update(config_override)
    
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT")
    print(f"{'='*60}")
    print(f"Environment: {exp_config['env_name']}")
    print(f"Norm: {exp_config['norm_type']} (epsilon={exp_config['epsilon']})")
    print(f"Players: {exp_config['num_players']}")
    print(f"Timesteps: {exp_config['timesteps']}")
    print(f"Simulations: {num_simulations}")
    print(f"{'='*60}\n")
    
    all_results = []
    
    # Run simulations
    for i in range(num_simulations):
        print(f"Running simulation {i+1}/{num_simulations}...", end=" ", flush=True)
        
        # Create config for this run using dictionary unpacking
        config = SimulationConfig(
            **exp_config,  # Unpack all config values (base + overrides)
            seed=seed_start + i if use_seeds else None  # Add seed if needed
        )
        
        # Run simulation
        result = run_simulation(config)
        all_results.append(result)
        
        final_welfare = result["cumulative_welfare"][-1]
        print(f"✓ (Final welfare: {final_welfare:.2f})")
    
    # Calculate summary statistics
    final_welfares = [r["cumulative_welfare"][-1] for r in all_results]
    mean_welfare = np.mean(final_welfares)
    std_welfare = np.std(final_welfares)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Mean final welfare: {mean_welfare:.2f} ± {std_welfare:.2f}")
    print(f"Min: {np.min(final_welfares):.2f}, Max: {np.max(final_welfares):.2f}")
    print(f"{'='*60}\n")
    
    # Save all data (update experiment name to include num_players if different)
    experiment_name = create_experiment_name()
    # Add num_players to experiment name if it's not the default 5
    if exp_config['num_players'] != 5:
        parts = experiment_name.rsplit('_', 1)  # Split off timestamp
        experiment_name = f"{parts[0]}_n{exp_config['num_players']}_{parts[1]}"
    
    save_experiment_data(experiment_name, all_results, config=exp_config)
    
    return all_results, experiment_name


if __name__ == "__main__":
    results, exp_name = run_experiment()
    print(f"Experiment '{exp_name}' completed successfully!")
    print(f"\nNext steps:")
    print(f"  1. Run dataprocess.py to visualize results")
    print(f"  2. Find your data in: data/{exp_name}/")

