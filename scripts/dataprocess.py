# Section 0: Standard library imports
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Section 1: Load experiment data
def load_experiment_data(experiment_name: str):
    """
    Load all data from a completed experiment.
    
    Args:
        experiment_name: Name of the experiment folder in data/
        
    Returns:
        Dictionary containing loaded data and metadata
    """
    data_dir = ROOT_DIR / "data" / experiment_name
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Experiment '{experiment_name}' not found in data/")
    
    # Load configuration
    config_file = data_dir / "experiment_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load cumulative welfare data
    cumulative_file = data_dir / "cumulative_welfare_all_runs.npy"
    cumulative_data = np.load(cumulative_file)
    
    # Load individual runs
    individual_runs = []
    run_files = sorted(data_dir.glob("run_*.npy"))
    for run_file in run_files:
        individual_runs.append(np.load(run_file))
    
    return {
        "config": config,
        "cumulative_welfare": cumulative_data,
        "individual_runs": individual_runs,
        "experiment_name": experiment_name
    }

# Section 2: Plot experiment results
def plot_experiment_results(data: dict, save_fig: bool = True):
    """
    Create visualizations for experiment results.
    
    Args:
        data: Dictionary from load_experiment_data()
        save_fig: If True, save plots to the experiment folder
    """
    config = data["config"]
    cumulative_data = data["cumulative_welfare"]
    individual_runs = data["individual_runs"]
    
    # Calculate statistics
    mean_cumulative = np.mean(cumulative_data, axis=0)
    std_cumulative = np.std(cumulative_data, axis=0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Cumulative welfare with confidence bands
    ax1 = axes[0]
    timesteps = np.arange(len(mean_cumulative))
    
    # Plot individual runs (transparent)
    for run in cumulative_data:
        ax1.plot(timesteps, run, alpha=0.2, color='gray', linewidth=0.5)
    
    # Plot mean with std band
    ax1.plot(timesteps, mean_cumulative, color='blue', linewidth=2, label='Mean')
    ax1.fill_between(
        timesteps, 
        mean_cumulative - std_cumulative, 
        mean_cumulative + std_cumulative,
        alpha=0.3, 
        color='blue',
        label='±1 std'
    )
    
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Cumulative Social Welfare', fontsize=12)
    ax1.set_title(
        f'Cumulative Welfare: {config["config"]["norm_type"]} (ε={config["config"]["epsilon"]})',
        fontsize=14
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Instantaneous rewards (mean ± std)
    ax2 = axes[1]
    individual_array = np.array(individual_runs)
    mean_rewards = np.mean(individual_array, axis=0)
    std_rewards = np.std(individual_array, axis=0)
    
    ax2.plot(timesteps, mean_rewards, color='green', linewidth=2, label='Mean')
    ax2.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        color='green',
        label='±1 std'
    )
    
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Reward per Step', fontsize=12)
    ax2.set_title('Instantaneous Team Reward', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        save_path = ROOT_DIR / "data" / data["experiment_name"] / "results_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def list_experiments():
    """List all available experiments in the data/ folder."""
    data_dir = ROOT_DIR / "data"
    
    if not data_dir.exists():
        print("No data/ folder found. Run an experiment first!")
        return []
    
    experiments = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    if not experiments:
        print("No experiments found in data/. Run experiment.py first!")
        return []
    
    print(f"\n{'='*60}")
    print("AVAILABLE EXPERIMENTS:")
    print(f"{'='*60}")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp}")
    print(f"{'='*60}\n")
    
    return experiments


if __name__ == "__main__":
    # List available experiments
    experiments = list_experiments()
    
    if not experiments:
        print("Run experiment.py first to generate data!")
    else:
        # Interactive selection
        print("Which experiment do you want to analyze?")
        print("  • Enter a number (1, 2, 3...)")
        print("  • Or type 'latest' for the most recent")
        print()
        
        choice = input("Your choice: ").strip()
        
        # Handle user input
        selected_experiment = None
        
        if choice.lower() == 'latest':
            selected_experiment = experiments[-1]
            print(f"\n→ Loading latest experiment: {selected_experiment}\n")
        elif choice.isdigit():
            idx = int(choice) - 1  # User sees 1-indexed list
            if 0 <= idx < len(experiments):
                selected_experiment = experiments[idx]
                print(f"\n→ Loading experiment {choice}: {selected_experiment}\n")
            else:
                print(f"\n❌ Error: Choice {choice} is out of range (1-{len(experiments)})")
                sys.exit(1)
        else:
            print(f"\n❌ Error: Invalid input '{choice}'. Please enter a number or 'latest'")
            sys.exit(1)
        
        # Load and visualize the selected experiment
        data = load_experiment_data(selected_experiment)
        print(f"Loaded {len(data['individual_runs'])} runs")
        print(f"Timesteps: {len(data['individual_runs'][0])}")
        print(f"\nGenerating plots...")
        
        plot_experiment_results(data, save_fig=True)

