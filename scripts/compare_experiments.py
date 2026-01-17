import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

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

    # Load cumulative welfare data (shape: [num_runs, timesteps])
    cumulative_file = data_dir / "cumulative_welfare_all_runs.npy"
    cumulative_data = np.load(cumulative_file)

    return {
        "config": config,
        "cumulative_welfare": cumulative_data,
        "experiment_name": experiment_name
    }


def compute_mean_std_cumulative(cumulative_runs: np.ndarray, per_capita: bool = False, num_players: int = 1):
    """Return mean and std across runs for cumulative welfare."""
    mean_curve = np.mean(cumulative_runs, axis=0)
    std_curve = np.std(cumulative_runs, axis=0)
    if per_capita and num_players > 0:
        mean_curve = mean_curve / num_players
        std_curve = std_curve / num_players
    return mean_curve, std_curve


def overlay_plot(experiment_names, shade_std: bool = True, save_path: Path | None = None, per_capita: bool = False):
    """Plot mean cumulative welfare curves for multiple experiments on one figure.

    Args:
        experiment_names: list of experiment directory names under data/
        shade_std: whether to draw ±1 std fill for each curve
        save_path: optional path to save the figure; if None, saves to data/results_overlay_<timestamp>.png
    """
    datasets = []
    for exp in experiment_names:
        data = load_experiment_data(exp)
        cfg = data["config"]["config"]
        num_players = int(cfg.get("num_players", 1))
        mean_curve, std_curve = compute_mean_std_cumulative(
            data["cumulative_welfare"],
            per_capita=per_capita,
            num_players=num_players
        )
        label = f"{cfg['env_name']} | {cfg['norm_type']} (ε={cfg['epsilon']}, n={num_players})"
        datasets.append({
            "label": label,
            "mean": mean_curve,
            "std": std_curve,
            "timesteps": np.arange(len(mean_curve)),
            "name": data["experiment_name"],
        })

    if not datasets:
        print("No experiments provided.")
        return

    # Align by shortest length
    min_len = min(len(d["mean"]) for d in datasets)
    for d in datasets:
        d["mean"] = d["mean"][:min_len]
        d["std"] = d["std"][:min_len]
        d["timesteps"] = d["timesteps"][:min_len]

    # Plot
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Distinct colors cycle
    color_cycle = plt.cm.tab10.colors

    for i, d in enumerate(datasets):
        color = color_cycle[i % len(color_cycle)]
        ax.plot(d["timesteps"], d["mean"], label=d["label"], color=color, linewidth=2)
        if shade_std:
            ax.fill_between(
                d["timesteps"],
                d["mean"] - d["std"],
                d["mean"] + d["std"],
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel("Timestep", fontsize=12)
    y_label = "Cumulative Reward per Capita" if per_capita else "Cumulative Social Welfare"
    title = "Per-Capita Cumulative Reward Comparison" if per_capita else "Cumulative Welfare Comparison Across Experiments"
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()

    # Save
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "per_capita_" if per_capita else ""
        save_path = ROOT_DIR / "data" / f"results_overlay_{suffix}{timestamp}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overlay plot saved to: {save_path}")

    # Optionally show
    plt.show()



def list_experiments():
    data_dir = ROOT_DIR / "data"
    if not data_dir.exists():
        return []
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir())


def main():
    parser = argparse.ArgumentParser(description="Overlay cumulative welfare for multiple experiments.")
    parser.add_argument(
        "experiments",
        nargs='*',
        help="Experiment directory names under data/. If omitted, will prompt a numbered list.")
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Disable ±1 std shading")
    parser.add_argument(
        "--per-capita",
        action="store_true",
        help="Plot cumulative reward per capita (divide by num_players)")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path for the overlay image")

    args = parser.parse_args()

    exps = args.experiments
    if not exps:
        all_exps = list_experiments()
        if not all_exps:
            print("No experiments found in data/. Run experiments first.")
            sys.exit(1)
        print(f"\n{'='*60}")
        print("AVAILABLE EXPERIMENTS:")
        print(f"{'='*60}")
        for i, exp in enumerate(all_exps, 1):
            print(f"  {i}. {exp}")
        print(f"{'='*60}\n")
        selection = input("Enter comma-separated numbers to select experiments (e.g., 1,3,4): ").strip()
        if not selection:
            print("No selection made.")
            sys.exit(1)
        try:
            indices = [int(s) - 1 for s in selection.split(',')]
            exps = [all_exps[i] for i in indices if 0 <= i < len(all_exps)]
        except ValueError:
            print("Invalid selection.")
            sys.exit(1)

    save_path = Path(args.save) if args.save else None
    overlay_plot(exps, shade_std=not args.no_std, save_path=save_path, per_capita=args.per_capita)


if __name__ == "__main__":
    main()
