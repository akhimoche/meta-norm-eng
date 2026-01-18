# Section 0: Standard library imports
import sys
import json
import time
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
    "norm_type": "sacred_apples",  # Options: "gpt5", "claude", "sacred_apples", etc., or "None"
    # Norm compliance (0.0 = always obey, 1.0 = always ignore)
    "epsilon": 0.2,   # Remember! 80% compliance is epsilon 0.2. This gets overridden by the sweep settings.
    "agent_type": "selfish",  # Options: "selfish" (more agent types coming soon)
    "num_players": 5,
    "timesteps": 500,  # Standard experiment length is 1000 timesteps.
}

# Experiment settings
num_simulations = 10  # Number of independent runs

# Sweep settings (optional)
# Options:
#   None         - Single experiment with base_config settings
#   "epsilon"    - Epsilon sweep for ONE norm (uses base_config["norm_type"])
#   "agent_count"- Agent count sweep for ONE norm
#   "norms"      - Epsilon sweep FOR EACH norm in sweep_norms (what you want!)
#                  This puts the epsilon sweep in a loop across all selected norms
sweep_mode = "norms"

# Epsilon values to test (used by "epsilon" and "norms" modes)
sweep_epsilons = [0.1, 0.2, 0.3]

# Agent counts to test (used by "agent_count" mode only)
sweep_agent_counts = [3, 5, 7]

# Norms to test (used by "norms" mode)
# When sweep_mode = "norms": runs epsilon sweep (sweep_epsilons) FOR EACH norm below
sweep_norms = [
    "claude_coreprotection",
    "claude_primecycles",
    "gemini_checkerboard",
    "gpt_coreprotection",
    # Add more norms here as needed
]

# Random seed settings (optional)
use_seeds = True  # Set to True if runs look too similar (adds reproducibility)
seed_start = 42  # Starting seed (will use 42, 43, 44... for each run)


# Section 2: Experiment execution
# Creates a descirptive folder name eg: 20260117_1343_commons_open_sacred_apples_n7_eps0p2_t500
def _short_env_name(env_name: str) -> str:
    """Shorten common env names for compact folder naming."""
    return (env_name
            .replace("commons_harvest__", "commons_")
            .replace("__", "_"))


def _format_epsilon(epsilon: float) -> str:
    """Format epsilon with a 'p' instead of '.' for filenames."""
    if epsilon is None:
        return "na"
    s = f"{epsilon:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def create_experiment_name(config: dict):
    """Generate a descriptive name for this experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    norm_str = config["norm_type"] if config["norm_type"] != "None" else "baseline"
    env_str = _short_env_name(config["env_name"])
    eps_str = f"eps{_format_epsilon(config['epsilon'])}"
    steps_str = f"t{config['timesteps']}"
    n_str = f"n{config['num_players']}"
    return f"{timestamp}_{env_str}_{norm_str}_{n_str}_{eps_str}_{steps_str}"


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
    data_dir = ROOT_DIR / "data" / "final experiments" / experiment_name
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


def _format_duration(seconds: float) -> str:
    """Format seconds into h/m for easier long-run tracking."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


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
    
    start_time = time.perf_counter()

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
        sim_start = time.perf_counter()
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
        sim_elapsed = time.perf_counter() - sim_start
        print(f"✓ (Final welfare: {final_welfare:.2f}, {_format_duration(sim_elapsed)})")
    
    # Calculate summary statistics
    final_welfares = [r["cumulative_welfare"][-1] for r in all_results]
    mean_welfare = np.mean(final_welfares)
    std_welfare = np.std(final_welfares)
    
    total_elapsed = time.perf_counter() - start_time
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Mean final welfare: {mean_welfare:.2f} ± {std_welfare:.2f}")
    print(f"Min: {np.min(final_welfares):.2f}, Max: {np.max(final_welfares):.2f}")
    print(f"Total elapsed: {_format_duration(total_elapsed)}")
    print(f"{'='*60}\n")
    
    # Save all data
    experiment_name = create_experiment_name(exp_config)
    
    save_experiment_data(experiment_name, all_results, config=exp_config)
    
    return all_results, experiment_name


def run_sweep(norm_override: str = None):
    """
    Run a sweep over agent counts or epsilon values.
    
    Args:
        norm_override: If provided, use this norm instead of base_config["norm_type"]
    """
    if sweep_mode not in {"agent_count", "epsilon", "norms"}:
        raise ValueError(f"Invalid sweep_mode: {sweep_mode}")

    # For norms sweep, this function is called with norm_override
    effective_norm = norm_override if norm_override else base_config["norm_type"]

    print(f"\n{'='*60}")
    print(f"STARTING SWEEP: epsilon (norm={effective_norm})")
    print(f"{'='*60}")

    if sweep_mode == "agent_count":
        sweep_values = sweep_agent_counts
        label = "num_players"
    else:  # epsilon or norms (norms uses epsilon sweep per norm)
        sweep_values = sweep_epsilons
        label = "epsilon"

    print(f"Testing across {len(sweep_values)} values: {sweep_values}")
    print(f"{'='*60}\n")

    all_experiments = {}
    for value in sweep_values:
        print(f"\nRunning with {label}={value}...")
        run_config = {label: value}
        if norm_override:
            run_config["norm_type"] = norm_override
        results, exp_name = run_experiment(config_override=run_config)
        all_experiments[value] = {
            "results": results,
            "experiment_name": exp_name
        }
        print(f"✓ Completed {label}={value}\n")

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE (norm={effective_norm})")
    print(f"{'='*60}")
    print(f"\nAll experiments completed:")
    for value in sweep_values:
        exp_name = all_experiments[value]["experiment_name"]
        final_welfares = [r["cumulative_welfare"][-1] for r in all_experiments[value]["results"]]
        mean_welfare = np.mean(final_welfares)
        std_welfare = np.std(final_welfares)
        print(f"  {label}={value}: {mean_welfare:.2f} ± {std_welfare:.2f} → {exp_name}")
    print(f"\n{'='*60}\n")

    return all_experiments


def run_norm_sweep():
    """
    Run epsilon sweeps across multiple norms.
    
    For each norm in sweep_norms, runs a full epsilon sweep using sweep_epsilons.
    This is equivalent to running experiment.py multiple times with different norms,
    but automated so you can set it up and let it run.
    """
    print(f"\n{'#'*60}")
    print(f"# MULTI-NORM SWEEP")
    print(f"{'#'*60}")
    print(f"Norms to test: {len(sweep_norms)}")
    for norm in sweep_norms:
        print(f"  • {norm}")
    print(f"Epsilon values per norm: {sweep_epsilons}")
    print(f"Total experiments: {len(sweep_norms) * len(sweep_epsilons)}")
    print(f"Simulations per experiment: {num_simulations}")
    print(f"Total simulations: {len(sweep_norms) * len(sweep_epsilons) * num_simulations}")
    print(f"{'#'*60}\n")
    
    all_norm_results = {}
    
    for i, norm in enumerate(sweep_norms):
        print(f"\n{'='*60}")
        print(f"NORM {i+1}/{len(sweep_norms)}: {norm}")
        print(f"{'='*60}")
        
        # Run epsilon sweep for this norm
        norm_experiments = run_sweep(norm_override=norm)
        all_norm_results[norm] = norm_experiments
        
        print(f"✓ Completed all epsilon values for {norm}\n")
    
    # Final summary
    print(f"\n{'#'*60}")
    print(f"# MULTI-NORM SWEEP COMPLETE")
    print(f"{'#'*60}")
    print(f"\nSummary by norm:")
    
    for norm in sweep_norms:
        print(f"\n  {norm}:")
        for eps in sweep_epsilons:
            if eps in all_norm_results[norm]:
                exp_data = all_norm_results[norm][eps]
                final_welfares = [r["cumulative_welfare"][-1] for r in exp_data["results"]]
                mean_welfare = np.mean(final_welfares)
                std_welfare = np.std(final_welfares)
                print(f"    ε={eps}: {mean_welfare:.2f} ± {std_welfare:.2f}")
    
    print(f"\n{'#'*60}\n")
    
    return all_norm_results


if __name__ == "__main__":
    if sweep_mode is None:
        results, exp_name = run_experiment()
        print(f"Experiment '{exp_name}' completed successfully!")
    elif sweep_mode == "norms":
        run_norm_sweep()
        print("Multi-norm sweep completed successfully!")
    else:
        run_sweep()
        print("Sweep completed successfully!")
    print(f"\nNext steps:")
    print(f"  1. Run dataprocess.py to visualize results")
    if sweep_mode is None:
        print(f"  2. Find your data in: data/{exp_name}/")
    else:
        print(f"  2. Find your data in: data/<experiment_name>/")

