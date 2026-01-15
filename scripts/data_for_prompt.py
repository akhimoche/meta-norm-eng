"""
Data summarizer for LLM prompts - extracts key metrics in copy-pasteable format.

Generates a simple performance summary that can be pasted directly into the
iterative improvement prompt.
"""

import sys
import json
from pathlib import Path

import numpy as np

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_experiment_data(experiment_name: str):
    """Load experiment data from the data folder."""
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
    
    return {
        "config": config,
        "cumulative_welfare": cumulative_data,
        "experiment_name": experiment_name
    }


def list_experiments():
    """List all available experiments."""
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


def generate_prompt_data(data: dict, sample_interval: int = 20):
    """
    Generate formatted performance data for LLM prompt.
    
    Args:
        data: Dictionary from load_experiment_data()
        sample_interval: Sample cumulative welfare every N timesteps
    """
    config = data["config"]
    cumulative_data = data["cumulative_welfare"]
    
    # Calculate statistics
    final_welfares = cumulative_data[:, -1]  # Last timestep of each run
    mean_final = np.mean(final_welfares)
    std_final = np.std(final_welfares)
    min_final = np.min(final_welfares)
    max_final = np.max(final_welfares)
    
    # Get mean cumulative curve
    mean_cumulative = np.mean(cumulative_data, axis=0)
    
    # Sample every N timesteps
    timesteps = len(mean_cumulative)
    sample_indices = list(range(0, timesteps, sample_interval))
    if sample_indices[-1] != timesteps - 1:  # Ensure we include the final timestep
        sample_indices.append(timesteps - 1)
    
    # Build output
    output = []
    output.append("**Performance Metrics:**\n")
    output.append(f"Final Cumulative Welfare ({config['config']['num_simulations']} runs):")
    output.append(f"  Mean: {mean_final:.2f} ± {std_final:.2f}")
    output.append(f"  Range: [{min_final:.2f}, {max_final:.2f}]\n")
    
    output.append(f"Mean Cumulative Welfare by Timestep (sampled every {sample_interval} steps):")
    for idx in sample_indices:
        output.append(f"  t={idx}: {mean_cumulative[idx]:.2f}")
    
    
    return "\n".join(output)


if __name__ == "__main__":
    # List available experiments
    experiments = list_experiments()
    
    if not experiments:
        sys.exit(1)
    
    # Interactive selection
    print("Which experiment do you want to summarize for the LLM prompt?")
    print("  • Enter a number (1, 2, 3...)")
    print("  • Or type 'latest' for the most recent")
    print()
    
    choice = input("Your choice: ").strip()
    
    # Handle user input
    selected_experiment = None
    
    if choice.lower() == 'latest':
        selected_experiment = experiments[-1]
        print(f"\n→ Selected latest experiment: {selected_experiment}\n")
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(experiments):
            selected_experiment = experiments[idx]
            print(f"\n→ Selected experiment {choice}: {selected_experiment}\n")
        else:
            print(f"\n❌ Error: Choice {choice} is out of range (1-{len(experiments)})")
            sys.exit(1)
    else:
        print(f"\n❌ Error: Invalid input '{choice}'. Please enter a number or 'latest'")
        sys.exit(1)
    
    # Load and generate summary
    data = load_experiment_data(selected_experiment)
    summary = generate_prompt_data(data, sample_interval=20)
    
    # Print the formatted output
    print("="*70)
    print("COPY-PASTE THIS INTO YOUR LLM PROMPT:")
    print("="*70)
    print()
    print(summary)
    print()
    print("="*70)
    print(f"\nExperiment: {selected_experiment}")
    print(f"Norm used: {data['config']['config']['norm_type']}")
    print("="*70)

