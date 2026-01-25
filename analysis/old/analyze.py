import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def aggregate_statistics_boxplots(
    algos,
    seeds=None,
    sut="real",
    llm_sut="gpt-5-chat",
    manual="initial",
    metrics=None,
    download_root="./wandb_download",
    output_dir="wandb_results",
):
    if metrics is None:
        metrics = [
            "total_tests",
            "num_failures",
            "average_distance",
            "num_warnings_violated",
            "failure_ratio",
        ]

    base_dir = f"{sut}_{llm_sut}_{manual}"
    base_path = os.path.join(download_root, base_dir)
    output_base_dir = os.path.join(output_dir, base_dir)
    os.makedirs(output_base_dir, exist_ok=True)  # create folder for plots/JSON
    print("base_path", base_path)
    # store per-seed metric values and filenames
    data = defaultdict(lambda: defaultdict(list))
    files_used = defaultdict(list)

    # optionally filter seeds
    all_seeds = seeds or os.listdir(base_path)

    # load evaluation_summary.json files
    for seed in all_seeds:
        seed_path = os.path.join(base_path, str(seed))
        if not os.path.isdir(seed_path):
            continue

        for algo in algos:
            algo_path = os.path.join(seed_path, algo)
            summary_file = os.path.join(algo_path, "evaluation_summary.json")
            if not os.path.isfile(summary_file):
                continue

            files_used[algo].append(summary_file)

            with open(summary_file, "r") as f:
                summary = json.load(f)

            for metric in metrics:
                if metric in summary:
                    data[algo][metric].append(summary[metric])

    # compute mean, std, n for JSON
    stats = {}
    for algo in algos:
        stats[algo] = {}
        for metric in metrics:
            values = data[algo].get(metric, [])
            if not values:
                continue
            mean_val = sum(values) / len(values)
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            stats[algo][metric] = {"mean": mean_val, "std": std_val, "n": len(values)}

    # include filenames in JSON
    stats["files_used"] = files_used

    # write JSON output
    stats_json_path = os.path.join(output_base_dir, "aggregated_statistics.json")
    with open(stats_json_path, "w") as f:
        json.dump(stats, f, indent=4)

    # define consistent color palette
    colors = plt.get_cmap("tab10").colors
    algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(algos)}

    # generate vertical boxplots
    for metric in metrics:
        algo_names = []
        metric_values = []
        box_colors = []

        for algo in algos:
            values = data[algo].get(metric, [])
            if values:
                algo_names.append(algo)
                metric_values.append(values)
                box_colors.append(algo_colors[algo])

        if not algo_names:
            continue

        plt.figure(figsize=(max(4, 0.6 * len(algo_names)), 4))
        box = plt.boxplot(metric_values, labels=algo_names, patch_artist=True,
                          vert=True, widths=0.6)

        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)

        plt.ylabel(metric.replace("_", " "))
        plt.xlabel("Test Generator")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout(pad=1.0)

        plot_path = os.path.join(output_base_dir, f"{metric}_boxplot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()


if __name__ == "__main__":
    algos = ["exida", "crisp", "atlas", "warnless", "smart"]
    seeds = [1, 2, 3, 4, 5, 6]
    aggregate_statistics_boxplots(
        algos=algos,
        seeds=seeds,
        sut="mock",
        llm_sut="gpt-4o-mini",
        manual="mini"
    )
