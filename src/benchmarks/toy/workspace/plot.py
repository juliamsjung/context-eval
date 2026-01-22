import json
from pathlib import Path
import matplotlib.pyplot as plt

WORKSPACE = Path(__file__).resolve().parent
RESULTS_PATH = WORKSPACE / "all_results.json"
PLOTS_PATH = WORKSPACE / "plots"

def load_results():
    """Load results from results.json."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} does not exist. Run the training script first.")
    
    with RESULTS_PATH.open() as f:
        results = json.load(f)
    
    return results

def plot_accuracy_vs_iterations(results):
    """Plot accuracy vs. iterations."""
    iterations = list(range(1, len(results) + 1))
    accuracies = [res["accuracy"] for res in results]

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, accuracies, marker="o", label="Accuracy")
    plt.title("Accuracy vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(iterations)
    plt.legend()
    plt.savefig(PLOTS_PATH / "accuracy_vs_iterations.png")
    plt.close()

def main():
    # Ensure the plots directory exists
    PLOTS_PATH.mkdir(exist_ok=True)

    # Load results
    results = load_results()

    # Generate plots
    plot_accuracy_vs_iterations(results)
    print(f"Plots saved to {PLOTS_PATH}")

if __name__ == "__main__":
    main()