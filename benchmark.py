import csv
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

from drone_simulation import DroneSimulation, START, GOAL, heuristic


TARGET_SUCCESSFUL_RUNS = 10
MAXIMUM_SEED_ATTEMPTS = 100
MAXIMUM_STEPS_PER_RUN = 500


def execute_benchmark():
    """Run multiple simulation trials using different random seeds."""
    successful_results = []
    failed_seeds = []

    seed = 42
    attempts = 0

    while (
        len(successful_results) < TARGET_SUCCESSFUL_RUNS
        and attempts < MAXIMUM_SEED_ATTEMPTS
    ):
        attempts += 1
        print(f"\nRunning benchmark trial with seed {seed}...")

        try:
            simulation = DroneSimulation(
                random_seed=seed,
                save_outputs=False,
            )
        except RuntimeError:
            print(f"Seed {seed}: no valid initial route.")
            failed_seeds.append(seed)
            seed += 1
            continue

        trial_start = time.perf_counter()

        while (
            not simulation.completed
            and simulation.step_count < MAXIMUM_STEPS_PER_RUN
        ):
            simulation.update_frame()

        benchmark_runtime = time.perf_counter() - trial_start

        if not simulation.completed:
            print(f"Seed {seed}: simulation did not complete.")
            failed_seeds.append(seed)
            seed += 1
            continue

        optimal_distance = heuristic(START, GOAL)
        actual_distance = len(simulation.trajectory) - 1
        efficiency = (
            optimal_distance / max(1, actual_distance)
        ) * 100

        successful_results.append(
            {
                "trial": len(successful_results) + 1,
                "seed": seed,
                "total_steps": simulation.step_count,
                "reroutes": simulation.reroute_count,
                "hazard_pauses": simulation.pause_count,
                "optimal_distance": optimal_distance,
                "actual_distance": actual_distance,
                "path_efficiency_percent": round(efficiency, 2),
                "runtime_seconds": round(benchmark_runtime, 6),
            }
        )

        seed += 1

    if not successful_results:
        raise RuntimeError("No benchmark trials completed successfully.")

    save_results(successful_results)
    create_benchmark_chart(successful_results)
    print_summary(successful_results, failed_seeds)


def save_results(results):
    """Save every successful benchmark trial to CSV."""
    results_directory = Path("results")
    results_directory.mkdir(parents=True, exist_ok=True)

    csv_path = results_directory / "benchmark_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBenchmark CSV saved to: {csv_path}")


def create_benchmark_chart(results):
    """Create charts comparing all benchmark trials."""
    graphs_directory = Path("evidence") / "graphs"
    graphs_directory.mkdir(parents=True, exist_ok=True)

    trial_numbers = [result["trial"] for result in results]
    steps = [result["total_steps"] for result in results]
    reroutes = [result["reroutes"] for result in results]
    efficiencies = [
        result["path_efficiency_percent"] for result in results
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(
        trial_numbers,
        steps,
        color="#0066CC",
        marker="o",
        linewidth=2,
    )
    axes[0].set_title("Steps per Trial")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Total steps")
    axes[0].set_xticks(trial_numbers)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].bar(
        trial_numbers,
        reroutes,
        color="#FF8800",
    )
    axes[1].set_title("Reroutes per Trial")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Reroute count")
    axes[1].set_xticks(trial_numbers)

    axes[2].plot(
        trial_numbers,
        efficiencies,
        color="#00AA44",
        marker="s",
        linewidth=2,
    )
    axes[2].set_title("Path Efficiency per Trial")
    axes[2].set_xlabel("Trial")
    axes[2].set_ylabel("Efficiency (%)")
    axes[2].set_xticks(trial_numbers)
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    figure.suptitle(
        "Autonomous Drone Multi-Trial Benchmark",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        graphs_directory / "benchmark_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(
        "Benchmark chart saved to: "
        f"{graphs_directory / 'benchmark_summary.png'}"
    )


def print_summary(results, failed_seeds):
    """Print aggregate benchmark statistics."""
    steps = [result["total_steps"] for result in results]
    reroutes = [result["reroutes"] for result in results]
    pauses = [result["hazard_pauses"] for result in results]
    efficiencies = [
        result["path_efficiency_percent"] for result in results
    ]
    runtimes = [result["runtime_seconds"] for result in results]

    print("\n" + "=" * 55)
    print("MULTI-TRIAL BENCHMARK SUMMARY")
    print("=" * 55)
    print(f"Successful trials       : {len(results)}")
    print(f"Failed seeds            : {len(failed_seeds)}")
    print(f"Average steps           : {statistics.mean(steps):.2f}")
    print(f"Average reroutes        : {statistics.mean(reroutes):.2f}")
    print(f"Average hazard pauses   : {statistics.mean(pauses):.2f}")
    print(
        f"Average path efficiency : "
        f"{statistics.mean(efficiencies):.2f}%"
    )
    print(
        f"Efficiency std. dev.    : "
        f"{statistics.stdev(efficiencies):.2f}%"
    )
    print(
        f"Average algorithm time  : "
        f"{statistics.mean(runtimes):.6f} seconds"
    )
    print("=" * 55)


if __name__ == "__main__":
    execute_benchmark()