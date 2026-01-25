from analysis.download import download_run_artifacts_relative
from analysis.utils import AnalysisPlots, AnalysisException, AnalysisTables, AnalysisDiversity
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from collections import defaultdict
import pandas as pd
from typing import List, Tuple, Literal, Optional, Dict
import pickle
import tqdm
import seaborn as sns
from llm.utils.embeddings_local import get_embedding as get_embedding_local
from llm.utils.embeddings_openai import get_embedding as get_embedding_openai
from analysis.models import Utterance
from tqdm import tqdm

def filter_safety(runs):
    res = []
    take = False
    for run in runs:
        if run.id == "wqpwm2wj":
            take = True
        if take and run.state == "finished":
            res.append(run)
    return res


run_filters = {"SafeLLM": filter_safety}


def convert_name(run_name: str) -> str:
    # """
    # Convert a run name into a standardized SUT string by scanning each
    # word (separated by '_') in order and including all matching identifiers.
    # """
    # sut_keywords = {
    #     "ipa": "",
    #     "yelp": "",
    #     "gpt-4o": "GPT-4o",
    #     "gpt-5-chat": "GPT-5-Chat",
    #     "deepseek-v3-0324": "DeepSeek-V3",
    #     "chatbmw": "",
    #     "mistral": "Mistral-7B",
    #     "qwen3" : "Qwen3-8B",
    #     "deepseek-v2" : "DeepSeek-V2-16B"
    # }

    # words = run_name.split("_")
    # sut_parts = []
    # for w in words:
    #     w_lower = w.lower()
    #     if w_lower in sut_keywords:
    #         print(w_lower)
    #         kwd = sut_keywords[w_lower]
    #         if kwd != "":
    #             sut_parts.append(kwd)
    
    # return "_".join(sut_parts) if sut_parts else "unknown_sut"
    return run_name.replace("_"," ").lower().replace("real","advanced").replace("initial","manual1").replace("mini","manual2")

metric_names = {
    "failures": "Number of Failures",
    "critical_ratio": "Critical Ratio",
}

algo_names_map = {
    "crisp": "crisp",
    "exida": "exida",
    "smart": "smart",
    "warnless": "warnless",
    "atlas": "atlas"
}

algorithms = [
    "crisp",
    "exida",
    "smart",
    "warnless",
    "atlas"
]


def get_algo_name(algo, features):
    algo_name = algo_names_map.get((algo, features), None)
    if algo_name is None:
        algo_name = algo_names_map[algo]
    return algo_name


def get_real_tests(path_name:str) -> Tuple[int,int]:
    all_tests = os.path.join(path_name, "evaluation_summary.json")

    # Load the JSON data
    with open(all_tests, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure data is iterable (e.g., a list of objects)
    if isinstance(data, dict):
        data = [data]

    num_failures = int(get_summary(path_name)["num_failures"]) 
    total_tests = int(get_summary(path_name)["total_tests"])
    num_warnings_violated = int(get_summary(path_name)["num_warnings_violated"])

    # print(f"num_real_tests: {num_real_tests}")
    # print(f"num_real_fail: {num_real_fail}")
    return total_tests, num_failures, num_warnings_violated

def get_embeddings(
    artifact_directory_path: str,
    local: bool = False,
    input: bool = True
) -> Tuple[List[str], np.ndarray]:
    """
    Extract embeddings for 'request' or 'answer' fields from evaluated_tests.json.

    :param artifact_directory_path: Path to experiment folder containing evaluated_tests.json
    :param local: Use local embedding function if True, else OpenAI API
    :param input: If True, embed 'request' field; else embed 'answer' field
    :return: Tuple of list of texts and np.ndarray of embeddings
    """
    file_name = "request_embeddings.pkl" if input else "answer_embeddings.pkl"
    pickle_path = os.path.join(artifact_directory_path, file_name)

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    json_path = os.path.join(artifact_directory_path, "evaluated_tests.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist")

    # Load JSON
    with open(json_path, "r", encoding="utf8") as f:
        data = json.load(f)

    # Extract texts
    texts = []
    for item in data:
        if not item.get("is_valid", True):
            continue

        if input:
            # Use request field
            request = item.get("test_case", {}).get("request", None)
            if request:
                texts.append(request)
        else:
            # Use answer field
            answer = item.get("system_response", {}).get("answer", None)
            if answer:
                texts.append(answer)
    if len(texts) == 0:
        raise ValueError(f"No valid texts found in {json_path} for input={input}")

    # Select embedding function
    get_embedding = get_embedding_local if local else get_embedding_openai

    # Compute embeddings
    embeddings = []
    for text in tqdm(texts, desc="Computing embeddings"):
        emb = np.array(get_embedding(text)).reshape(1, -1)
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)  # shape: [num_texts, embedding_dim]    embeddings = np.concatenate(embeddings, axis=0)  # shape: [num_texts, embedding_dim]

    # Cache results
    with open(pickle_path, "wb") as f:
        pickle.dump((texts, embeddings), f)

    return texts, embeddings


def get_run_history_table(run_path: str, freq: str = "1min", th_content = None, th_response = None):
    history_file = os.path.join(run_path, "run_history.csv")
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"No run_history.csv found in {run_path}")

    df = pd.read_csv(history_file)

    # Handle timestamps
    if "timestamp" not in df.columns and "_timestamp" in df.columns:
        df["timestamp"] = df["_timestamp"]

    # Convert to datetime
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.drop_duplicates(subset="time").sort_values("time")

    # --- Normalize so each run starts at t=0 ---
    df["time"] = df["time"] - df["time"].iloc[0]

    # Set normalized time as index
    df = df.set_index("time")

    # Resample and interpolate on a uniform grid
    if isinstance(df.index, pd.TimedeltaIndex):
        df = df.resample(freq).first()
        df = df.interpolate(method="time")

    # interpolate based on real failures

    num_tests_all, num_real_fail = get_real_tests(path_name = run_path)
    
    all_fail = df["failures"].iloc[-1]

    # print(f"per cent real fail: ", num_real_fail/all_fail)
    # print(f"ratio_real:", num_real_fail/num_tests_all)

    df["failures"] = df["failures"] * num_real_fail/all_fail
    df["critical_ratio"] = df.critical_ratio * num_real_fail/all_fail

    return df

def diversity_report(
        algorithms,
        project="SafeLLM",
        output_path="diversity",
        input: bool = True,
        max_num_clusters = 150,
        silhouette_threshold = 20,
        visualize: bool = True,
        mode: Literal["separated", "merged"] = "separated",
        num_seeds: Optional[int] = None,
        local_root: str = None,
        run_filters = None,
        save_folder= None
    ):
    if mode == "merged" and num_seeds is None:
        raise AnalysisException("Provide the number of seed in merged mode")
    output_path = os.path.join(output_path, "input" if input else "output")


    if local_root != None:
        artifact_paths = download_run_artifacts_relative(f"opentest/{project}", 
                                                         local_root=local_root,
                                                         filter_runs = run_filters[project])
    else:
        artifact_paths = download_run_artifacts(f"opentest/{project}", run_filters[project])
    
    print(f"Applying diversity analysis for {len(artifact_paths)} runs.")

    cached_embeddings = dict()
    suts = []
    result = {}
    print("output_path:", output_path)
    if os.path.exists(os.path.join(output_path, "report.json")):
        with open(os.path.join(output_path, "report.json"), "r") as f:
            result = json.load(f)
    else:
        for i, (sut, algos) in enumerate(artifact_paths.items()):
            suts.append(sut)
            print(sut)
            result[sut] = dict()
            algo_names = [get_algo_name(*key) for key in algos.keys()]

            for algo_name, paths in zip(algo_names, algos.values()):
                avg_max_distances = []
                avg_distances = []
                for path in paths:
                    _, embeddings = get_embeddings(path)
                    cached_embeddings[path] = embeddings
                    avg_max_distances.append(AnalysisDiversity.average_max_distance(embeddings))
                    avg_distances.append(AnalysisDiversity.average_distance(embeddings))
                result[sut][algo_name] = {
                    "avg_max_distance": avg_max_distances,
                    "avg_distance": avg_distances,
                }
            for nm in algo_names:
                result[sut][nm]["coverage"] = []
                result[sut][nm]["entropy"] = []

            if mode == "separated":
                original_seeds = min([len(paths) for paths in algos.values()])
                if num_seeds is None:
                    num_seeds = original_seeds
                for seed in range(num_seeds):
                    algo_counts = dict()
                    to_cluster = []
                    for algo_name, paths in zip(algo_names, algos.values()):
                        embeddings = cached_embeddings[paths[seed % original_seeds]]
                        algo_counts[algo_name] = embeddings.shape[0]
                        to_cluster.append(embeddings)
                    to_cluster = np.concatenate(to_cluster)
                    max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
                    clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
                        data=to_cluster,
                        n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
                        seed=seed,
                        silhouette_threshold=silhouette_threshold,
                    )
                    coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
                        labels, centers, algo_names, algo_counts,
                    )
                    for nm in coverage:
                        result[sut][nm]["coverage"].append(coverage[nm])
                        result[sut][nm]["entropy"].append(entropy[nm])
                    os.makedirs(os.path.join(output_path, sut), exist_ok=True)
                    cluster_data = (
                        to_cluster,
                        algo_names,
                        algo_counts,
                        labels,
                        centers,
                        seed,
                    )
                    with open(os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl"), "wb") as f:
                        pickle.dump(cluster_data, f)
                    if visualize:
                        AnalysisPlots.plot_clusters(
                            to_cluster,
                            centers,
                            os.path.join(output_path, sut, f"clusters_seed{seed}"),
                            algo_names,
                            algo_counts,
                            seed=seed,
                        )
            elif mode == "merged":
                algo_counts = defaultdict(int)
                to_cluster = []
                for algo_name, paths in tqdm.tqdm(zip(algo_names, algos.values())):
                    for path in paths:
                        embeddings = cached_embeddings[path]
                        algo_counts[algo_name] += embeddings.shape[0]
                        to_cluster.append(embeddings)
                to_cluster = np.concatenate(to_cluster)
                max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
                for seed in range(num_seeds):
                    clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
                        data=to_cluster,
                        n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
                        seed=seed,
                        silhouette_threshold=silhouette_threshold,
                    )
                    coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
                        labels, centers, algo_names, algo_counts,
                    )
                    for nm in coverage:
                        result[sut][nm]["coverage"].append(coverage[nm])
                        result[sut][nm]["entropy"].append(entropy[nm])
                    os.makedirs(os.path.join(output_path, sut), exist_ok=True)   
                    cluster_data = (
                        to_cluster,
                        algo_names,
                        algo_counts,
                        labels,
                        centers,
                        seed,
                    )
                    with open(os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl"), "wb") as f:
                        pickle.dump(cluster_data, f)
                    if visualize:
                        AnalysisPlots.plot_clusters(
                            to_cluster,
                            centers,
                            os.path.join(output_path, sut, f"clusters_seed{seed}"),
                            algo_names,
                            algo_counts,
                            seed=seed,
                        )
            else:
                raise AnalysisException("Unknown mode")
            print(result)

        with open(os.path.join(output_path, "report.json"), "w") as f:
            json.dump(result, f)

    data_dict = defaultdict(list)
    suts = list(artifact_paths.keys())
    for algorithm in algorithms:
        data_dict["Algorithm"].append(algorithm)
        for sut in suts:
            for metric in [
                "avg_max_distance",
                "avg_distance",
                "coverage",
                "entropy",
            ]:
                if algorithm not in result[sut]:
                    mean = None
                else:
                    values = result[sut][algorithm].get(metric, None)
                    mean = np.mean(values) if values is not None else None
                data_dict[f"{sut}.{metric}"].append(mean)
    pd.DataFrame(data_dict).to_csv(os.path.join(output_path, "scores.csv"), index=False)

    
    for metric in [
                "avg_max_distance",
                "avg_distance",
                "coverage",
                "entropy",
            ]:
        statistics = {}
        for sut in suts:
            algo_names = list(result[sut].keys())
            values = [result[sut][algo][metric] for algo in algo_names]
            statistics[sut] = AnalysisTables.statistics(values, algo_names)
        data_dict = defaultdict(list)
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                data_dict["Algorithm 1"].append(algorithms[i])
                data_dict["Algorithm 2"].append(algorithms[j])
                for sut in suts:
                    stats = statistics[sut][algorithms[i]][algorithms[j]]
                    if len(stats) > 0:
                        data_dict[f"{sut.capitalize()}.P-Value"].append(stats[0])
                        data_dict[f"{sut.capitalize()}.Effect Size"].append(stats[1])
                    else:
                        data_dict[f"{sut.capitalize()}.P-Value"].append(None)
                        data_dict[f"{sut.capitalize()}.Effect Size"].append(None)
        df = pd.DataFrame(data_dict)
        df = df.dropna()
        df.to_csv(os.path.join(output_path, save_folder, f"{metric}_stats.csv"), index=False)        
    return result

def get_summary(artifact_directory_path: str):
    json_path = os.path.join(artifact_directory_path, "evaluation_summary.json")
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary

def statistics_table(
    algorithms,
    project="SafeLLM",
    metric="failures",
    path="table.csv",
    run_filters = None,
    one_per_name: bool = False,
    experiments_folder: str = "wandb_download"
):     
    save_path = path
    if run_filters is not None and project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts_relative(local_root=experiments_folder, 
                                                     wb_project_path=f"opentest/{project}",
                                                     filter_runs=run_filters[project],
                                                     one_per_name=one_per_name)
    statistics = {}
    suts = []
    
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        suts.append(sut)
        algo_names = [get_algo_name(*key) for key in algos.keys()]
        print("algos:", algo_names)
        if metric == "failures":
            # Compute number of failures for each path
            values = [
                [get_real_tests(path)[1] for path in paths]  # index 1 = num_real_fail
                for paths in algos.values()
            ]
            print("failures:", values)
        elif metric == "critical_ratio":
            # Compute ratio = num_real_fail / num_real_tests for each path
            values = []
            for paths in algos.values():
                algo_values = []
                for path in paths:
                    num_real_tests, num_real_fail = get_real_tests(path)
                    ratio = (
                        num_real_fail / num_real_tests
                        if num_real_tests > 0
                        else 0.0
                    )
                    algo_values.append(ratio)
                values.append(algo_values)
            print("critical_ratio:", values)

        else:
            raise AnalysisException(f"Unknown metric: {metric}")
        statistics[sut] = AnalysisTables.statistics(values, algo_names)

    data = defaultdict(list)
    print("algorithms: ", algorithms)
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            data["Algorithm 1"].append(algorithms[i])
            data["Algorithm 2"].append(algorithms[j])
            for sut in suts:
                stats = statistics[sut][algorithms[i]][algorithms[j]]
                if len(stats) > 0:
                    data[f"{sut.capitalize()}.P-Value"].append(stats[0])
                    data[f"{sut.capitalize()}.Effect Size"].append(stats[1])
                else:
                    data[f"{sut.capitalize()}.P-Value"].append(None)
                    data[f"{sut.capitalize()}.Effect Size"].append(None)
                print(stats)
    df = pd.DataFrame(data)


    print(f"Before dropna: {len(df)} rows")

    # Drop only fully empty rows
    df = df.dropna(how="all")
    print(f"After dropna: {len(df)} rows")


    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)
    
def boxplots_multi_threshold(
    algorithms,
    project="SafeLLM",
    size=(18, 6),
    metrics=("failures", "critical_ratio"),
    file_name="plot",
    experiments_folder="wandb_download",
    run_filters=None,
    one_per_name=False,
    thresholds=None,   # list of thresholds, used for both th_content and th_response
):
    if project not in run_filters:
        raise AnalysisException(
            "Please implement runs filter for your project in opensbt.visualization.llm_figures"
        )

    if not thresholds or len(thresholds) == 0:
        raise ValueError("Please provide at least one threshold value in 'thresholds'.")

    all_results = []  # store results for all thresholds, SUTs, algorithms, and metrics

    # === Iterate over thresholds ===
    for th in thresholds:
        print(f"\n=== Processing threshold = {th} ===")

        artifact_paths = download_run_artifacts_relative(
            f"opentest/{project}",
            local_root=experiments_folder,
            filter_runs=run_filters[project],
            one_per_name=one_per_name
        )

        for metric in metrics:
            for sut, algos in artifact_paths.items():
                for (algo, features), paths in algos.items():
                    algo_name = get_algo_name(algo, features)
                    values = []

                    for path in paths:
                        df = get_run_history_table(path, th_content=th, th_response=th)
                        if metric not in df.columns:
                            continue
                        values.extend(df[metric].dropna().tolist())

                    if not values:
                        continue

                    all_results.append({
                        "SUT": sut,
                        "Algorithm": algo_name,
                        "Threshold": th,
                        "Metric": metric,
                        "mean": np.mean(values),
                        "std": np.std(values),
                    })

    if not all_results:
        raise RuntimeError("No data collected for any threshold or metric.")

    stats_df = pd.DataFrame(all_results)
    stats_df["mean_std"] = stats_df.apply(
        lambda r: f"{r['mean']:.3f} ({r['std']:.3f})", axis=1
    )

    # === Create one combined summary table per SUT and metric ===
    print("\n=== Combined Summary Tables ===")
    for sut in stats_df["SUT"].unique():
        for metric in stats_df["Metric"].unique():
            subset = stats_df[(stats_df["SUT"] == sut) & (stats_df["Metric"] == metric)]
            if subset.empty:
                continue

            pivot = subset.pivot_table(
                index="Algorithm",
                columns="Threshold",
                values="mean_std",
                aggfunc="first"
            ).reset_index()

            print(f"\n--- {sut} | {metric} ---")
            print(pivot.to_string(index=False))

            # Save to CSV
            csv_path = file_name.replace(".png", f"_{sut}_{metric}_summary.csv")
            pivot.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")
            
def boxplots(
    algorithms,
    project="SafeLLM",
    size=(18, 6),
    metric="failures",
    file_name="plot",
    experiments_folder="wandb_download",
    run_filters=None,
    one_per_name=False,
    th_content=None,
    th_response=None
):
    if project not in run_filters:
        raise AnalysisException(
            "Please implement runs filter for your project in opensbt.visualization.llm_figures"
        )

    artifact_paths = download_run_artifacts_relative(
        f"opentest/{project}",
        local_root=experiments_folder,
        filter_runs=run_filters[project],
        one_per_name=one_per_name
    )

    if metric == "critical_ratio":
        fig, axes = plt.subplots(1, len(artifact_paths), sharey="row")
    else:
        fig, axes = plt.subplots(1, len(artifact_paths))
    fig.set_size_inches(*size)

    if len(artifact_paths) == 1:
        axes = [axes]

    ticks_kwargs = {}
    if metric == "critical_ratio":
        ticks_kwargs["yticks"] = np.linspace(0.0, 1.0, 6)
        ticks_kwargs["ylim"] = (0.0, 1.0)
        plt.setp(axes, **ticks_kwargs)

    # Collect all values for descriptive statistics
    all_stats = []

    for i, (sut, algos) in enumerate(artifact_paths.items()):
        name_to_dfs = {}
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [
                get_run_history_table(path, th_content=th_content, th_response=th_response)
                for path in paths
            ]
            name_to_dfs[algo_name] = dfs

        algo_names = []
        dfs_list = []
        for algorithm in algorithms:
            if algorithm in name_to_dfs:
                algo_names.append(algorithm)
                dfs_list.append(name_to_dfs[algorithm])

        # Extract values for statistics
        for algo_name, dfs in zip(algo_names, dfs_list):
            values = []
            for df in dfs:
                if metric not in df.columns:
                    continue
                values.extend(df[metric].dropna().tolist())
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                median_val = np.median(values)
                count_val = len(values)
                all_stats.append({
                    "SUT": sut,
                    "Algorithm": algo_name,
                    "mean": mean_val,
                    "std": std_val,
                    "median": median_val,
                    "count": count_val
                })

        # Plotting
        AnalysisPlots.boxplot(axes[i], dfs_list, algo_names, metric=metric)
        # axes[i].set_title(convert_name(sut.capitalize()))
        axes[i].set_box_aspect(1)
        axes[i].set_xticklabels(algo_names, rotation='vertical')

    # Label y-axis for the first subplot only
    axes[0].set_ylabel(metric_names[metric], labelpad=10, fontsize=18)

    legend_handles = {
        label: plt.Line2D([0], [0], color=col, lw=10)
        for label, col in AnalysisPlots.label_colors.items()
    }

    fig.tight_layout()
    plt.savefig(file_name, format="pdf")

    # === Print and save summary statistics ===
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df = stats_df.sort_values(["SUT", "Algorithm"]).reset_index(drop=True)

        print("\n=== Summary statistics per SUT and Algorithm ===")
        for sut in stats_df["SUT"].unique():
            print(f"\n--- {sut} ---")
            sut_df = stats_df[stats_df["SUT"] == sut]
            for _, row in sut_df.iterrows():
                print(
                    f"{row['Algorithm']:<10} | mean = {row['mean']:.4f}, "
                    f"std = {row['std']:.4f}, median = {row['median']:.4f} (n={int(row['count'])})"
                )

        stats_path = file_name.replace(".png", "_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"\nStatistics saved to: {stats_path}")
    
def plot_boxplots_by_algorithm_raw(
    project: str = "SafeLLM",
    metric: str = "failures",
    run_filters=None,
    one_per_name: bool = False,
    experiments_folder: str = "wandb_download",
    save_path: str = "plots/boxplots_raw.png"
):
    """
    Create boxplots of raw run values for each algorithm (one subplot per algorithm).
    Uses the same artifact data structure as in last_values_table().
    """

    artifact_paths = download_run_artifacts_relative(
        local_root=experiments_folder,
        wb_project_path=f"opentest/{project}",
        filter_runs=run_filters[project] if run_filters else None,
        one_per_name=one_per_name,
    )

    all_data = []

    for sut, algos in artifact_paths.items():
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)

            # Collect per-run metric values
            values = []
            if metric == "failures":
                for path in paths:
                    values.append(get_real_tests(path)[1])  # number of failures
            elif metric == "num_warnings_violated":
                for path in paths:
                    values.append(get_real_tests(path)[2])  # number of failures
            elif metric == "critical_ratio":
                for path in paths:
                    num_real_tests, num_real_fail,_ = get_real_tests(path)
                    ratio = num_real_fail / num_real_tests if num_real_tests > 0 else 0.0
                    values.append(ratio)
            # elif metric in ("coverage", "entropy"):
            #     import json
            #     for path in paths:
            #         with open(path + f"/diversity/input/{sut}/report.json", "r") as f:
            #             report = json.load(f)
            #         # Flatten all seeds for the current SUT
            #         for seed_data in report.values():
            #             if sut in seed_data:
            #                 metric_values = seed_data[sut].get(metric, [])
            #                 values.extend(metric_values)
            else:
                raise AnalysisException(f"Unknown metric: {metric}")

            # Store all runs for that algorithm/SUT
            for v in values:
                all_data.append({
                    "SUT": sut,
                    "Algorithm": algo_name,
                    "Value": v
                })
    df = pd.DataFrame(all_data)
    if df.empty:
        raise RuntimeError("No data found to plot boxplots.")

    # Define consistent colors for algorithms
    algo_colors = {
        "smart": "#1f77b4",
        "crisp": "#b41fad",
        "warnless": "#ff7f0e",
        "exida": "#2ca02c",
        "atlas": "#999494"
    }

    suts = list(df["SUT"].unique())
    n_suts = len(suts)

    # # Define the desired full order
    # full_order = ["STELLAR", "T-wise", "Random"]

    # # Keep only the algorithms present in the dataset
    # algo_order = [algo for algo in full_order if algo in df["Algorithm"].unique()]
    # algo_order = full_order

    # Ensure the column is categorical with this order
    # df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=algo_order, ordered=True)

    # Create one subplot per SUT
    fig, axes = plt.subplots(1, n_suts, figsize=(5 * n_suts, 5), sharey=True)

    if n_suts == 1:
        axes = [axes]

    for ax, sut in zip(axes, suts):
        sut_df = df[df["SUT"] == sut]

        sns.boxplot(
            data=sut_df,
            x="Algorithm",
            y="Value",
            palette=algo_colors,
            ax=ax,
            width=0.6,
            # order=algo_order
        )

        ax.set_title(convert_name(sut), fontsize=17, weight="bold")  # title = model name
        ax.set_ylabel(metric.replace("_", " ").capitalize(), fontsize = 14)
        ax.set_xlabel("")  # remove x-axis label
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        # Style: gray background, white grid, no borders
        ax.set_facecolor("0.8")
        ax.grid(visible=True, which="both", color="white", linestyle="-", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, format="pdf")
    print(f"Boxplots saved to: {save_path}")
    plt.close()
    
def last_values_table(
    project: str = "SafeLLM",
    metrics: list | str = "failures",
    path: str = "table.csv",
    run_filters=None,
    one_per_name: bool = False,
    experiments_folder: str = "wandb_download",
):
    save_path = path

    if run_filters is not None and project not in run_filters:
        raise AnalysisException(
            "Please implement runs filter for your project in opensbt.visualization.llm_figures"
        )

    artifact_paths = download_run_artifacts_relative(
        local_root=experiments_folder,
        wb_project_path=f"opentest/{project}",
        filter_runs=run_filters[project] if run_filters else None,
        one_per_name=one_per_name,
    )

    if isinstance(metrics, str):
        metrics = [metrics]

    summary_data = []

    for sut, algos in artifact_paths.items():
        algo_names = [get_algo_name(*key) for key in algos.keys()]
        print("algos:", algo_names)

        for algo_name, paths in zip(algo_names, algos.values()):
            row = {"Algorithm": algo_name,"SUT": sut}

            for metric in metrics:
                if metric == "failures":
                    values = [get_real_tests(path)[1] for path in paths]
                elif metric == "num_warnings_violated":
                    values = [get_real_tests(path)[2] for path in paths]
                elif metric == "critical_ratio":
                    values = []
                    for path in paths:
                        num_real_tests, num_real_fail, _ = get_real_tests(path)
                        ratio = num_real_fail / num_real_tests if num_real_tests > 0 else 0.0
                        values.append(ratio)
                else:
                    raise AnalysisException(f"Unknown metric: {metric}")

                row[f"{metric}_mean"] = np.mean(values) if values else np.nan
                row[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0

            summary_data.append(row)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # algorithm_order = ["STELLAR", "Random", "T-wise"]
    summary_df = pd.DataFrame(summary_data)
    # summary_df["Algorithm"] = pd.Categorical(summary_df["Algorithm"], categories=algorithm_order, ordered=True)
    # summary_df = summary_df.sort_values(by=["SUT", "Algorithm"]).reset_index(drop=True)

    summary_df.to_csv(save_path, index=False)
    print(f"\nSummary table of mean/std saved to: {save_path}")

    # Determine LaTeX column alignment dynamically
    n_metrics = len(metrics)
    col_format = "ll" + "cc" * n_metrics  # 2 left + 2 per metric (mean/std)
    
    latex_path = os.path.splitext(save_path)[0] + ".tex"
    summary_df.to_latex(
        latex_path,
        index=False,
        float_format="%.3f",
        caption=f"Summary of metrics per SUT and algorithm",
        label="tab:metric_summary",
        column_format=col_format
    )
    print(f"Summary table exported to LaTeX: {latex_path}")

    normalized_df = summary_df.copy()

    for metric in metrics:
        mean_col = f"{metric}_mean"
        # do not normalize critical_ratio
        if metric == "critical_ratio":
            continue

        # normalize per SUT
        for sut in normalized_df["SUT"].unique():
            mask = normalized_df["SUT"] == sut
            max_value = normalized_df.loc[mask, mean_col].max()

            if max_value > 0 and not np.isnan(max_value):
                normalized_df.loc[mask, mean_col] = (
                    normalized_df.loc[mask, mean_col] / max_value
                )
            else:
                normalized_df.loc[mask, mean_col] = np.nan

    normalized_path = os.path.splitext(save_path)[0] + "_normalized.csv"
    normalized_df.to_csv(normalized_path, index=False)

    print(f"Normalized summary table saved to: {normalized_path}")

def count_file_paths(nested_dict):
    total = 0
    for value in nested_dict.values():
        if isinstance(value, dict):
            total += count_file_paths(value)  # recurse into nested dict
        elif isinstance(value, list):
            total += len(value)  # count the file paths in the list
    return total

def diversity_report(
        algorithms,
        project="SafeLLM",
        output_path="diversity",
        input: bool = True,
        max_num_clusters = 150,
        silhouette_threshold = 20,
        visualize: bool = True,
        mode: Literal["separated", "merged"] = "separated",
        num_seeds: Optional[int] = None,
        local_root: str = None,
        one_per_name: bool = True,
        run_filters: Dict = None
    ):
    if mode == "merged" and num_seeds is None:
        raise AnalysisException("Provide the number of seed in merged mode")
    output_path = os.path.join(output_path, "input" if input else "output")

    if local_root != None:
        artifact_paths = download_run_artifacts_relative(f"opentest/{project}", 
                                                            local_root=local_root,
                                                            filter_runs = run_filters[project],
                                                            one_per_name=one_per_name)
    else:
        artifact_paths = download_run_artifacts(f"opentest/{project}", run_filters[project])
    print(f"Applying diversity analysis for {count_file_paths(artifact_paths)} runs.")
    cached_embeddings = dict()
    suts = []
    result = {}
    if os.path.exists(os.path.join(output_path, "report.json")):
        with open(os.path.join(output_path, "report.json"), "r") as f:
            result = json.load(f)
    else:
        for i, (sut, algos) in enumerate(artifact_paths.items()):
            suts.append(sut)
            print(sut)
            result[sut] = dict()
            algo_names = [get_algo_name(*key) for key in algos.keys()]

            for algo_name, paths in zip(algo_names, algos.values()):
                avg_max_distances = []
                avg_distances = []
                for path in tqdm(paths, desc=f"Processing embeddings generation"):
                    _, embeddings = get_embeddings(path)
                    cached_embeddings[path] = embeddings
                    avg_max_distances.append(AnalysisDiversity.average_max_distance(embeddings))
                    avg_distances.append(AnalysisDiversity.average_distance(embeddings))
                result[sut][algo_name] = {
                    "avg_max_distance": avg_max_distances,
                    "avg_distance": avg_distances,
                }
            for nm in algo_names:
                result[sut][nm]["coverage"] = []
                result[sut][nm]["entropy"] = []

            if mode == "separated":
                original_seeds = min([len(paths) for paths in algos.values()])
                if num_seeds is None:
                    num_seeds = original_seeds
                for seed in range(num_seeds):
                    algo_counts = dict()
                    to_cluster = []
                    for algo_name, paths in zip(algo_names, algos.values()):
                        embeddings = cached_embeddings[paths[seed % original_seeds]]
                        algo_counts[algo_name] = embeddings.shape[0]
                        to_cluster.append(embeddings)
                    to_cluster = np.concatenate(to_cluster)
                    max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
                    clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
                        data=to_cluster,
                        n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
                        seed=seed,
                        silhouette_threshold=silhouette_threshold,
                    )
                    coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
                        labels, centers, algo_names, algo_counts,
                    )
                    for nm in coverage:
                        result[sut][nm]["coverage"].append(coverage[nm])
                        result[sut][nm]["entropy"].append(entropy[nm])
                    os.makedirs(os.path.join(output_path, sut), exist_ok=True)
                    cluster_data = (
                        to_cluster,
                        algo_names,
                        algo_counts,
                        labels,
                        centers,
                        seed,
                    )
                    with open(os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl"), "wb") as f:
                        pickle.dump(cluster_data, f)
                    if visualize:
                        AnalysisPlots.plot_clusters(
                            to_cluster,
                            centers,
                            os.path.join(output_path, sut, f"clusters_seed{seed}"),
                            algo_names,
                            algo_counts,
                            seed=seed,
                        )
            elif mode == "merged":
                algo_counts = defaultdict(int)
                to_cluster = []
                for algo_name, paths in tqdm(zip(algo_names, algos.values())):
                    for path in paths:
                        embeddings = cached_embeddings[path]
                        algo_counts[algo_name] += embeddings.shape[0]
                        to_cluster.append(embeddings)
                to_cluster = np.concatenate(to_cluster)
                max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
                for seed in range(num_seeds):
                    pickled_data_path = os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl")
                    
                    
                    if os.path.exists(pickled_data_path):
                        with open(pickled_data_path, "rb") as f:
                            content = pickle.load(f)
                            (
                                to_cluster,
                                algo_names,
                                algo_counts,
                                labels,
                                centers,
                                seed,
                            ) = content       
                    else:
                        clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
                            data=to_cluster,
                            n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
                            seed=seed,
                            silhouette_threshold=silhouette_threshold,
                        )
                        os.makedirs(os.path.join(output_path, sut), exist_ok=True)   
                        cluster_data = (
                            to_cluster,
                            algo_names,
                            algo_counts,
                            labels,
                            centers,
                            seed,
                        )
                        with open(pickled_data_path, "wb") as f:
                            pickle.dump(cluster_data, f)

                    coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
                        labels, centers, algo_names, algo_counts,
                    )
                    for nm in coverage:
                        result[sut][nm]["coverage"].append(coverage[nm])
                        result[sut][nm]["entropy"].append(entropy[nm])                   
                    if visualize:
                        AnalysisPlots.plot_clusters(
                            to_cluster,
                            centers,
                            os.path.join(output_path, sut, f"clusters_seed{seed}"),
                            algo_names,
                            algo_counts,
                            seed=seed,
                        )
            else:
                raise AnalysisException("Unknown mode")
            print(result)

        with open(os.path.join(output_path, "report.json"), "w") as f:
            json.dump(result, f)

    data_dict = defaultdict(list)
    suts = list(artifact_paths.keys())

    for algorithm in algorithms:
        data_dict["Algorithm"].append(algorithm)
        for sut in suts:
            for metric in [
                "avg_max_distance",
                "avg_distance",
                "coverage",
                "entropy",
            ]:
                if algorithm not in result[sut]:
                    mean = None
                else:
                    values = result[sut][algorithm].get(metric, None)
                    mean = np.mean(values) if values is not None else None
                data_dict[f"{sut}.{metric}"].append(mean)

    scores_df = pd.DataFrame(data_dict)
    scores_df.to_csv(os.path.join(output_path, "scores.csv"), index=False)

    # ----------------------------------
    # NEW: prepare data for box plots
    # ----------------------------------

    records = []

    for sut in suts:
        for algorithm in algorithms:
            if algorithm not in result[sut]:
                continue
            for metric in ["coverage", "entropy"]:
                values = result[sut][algorithm].get(metric, None)
                if values is None:
                    continue
                for v in values:
                    records.append({
                        "SUT": sut,
                        "Algorithm": algorithm,
                        "Metric": metric,
                        "Value": v,
                    })

    all_data = pd.DataFrame(records)

    df = pd.DataFrame(all_data)
    if df.empty:
        raise RuntimeError("No data found to plot boxplots.")

    # ----------------------------------
    # Style configuration (same as provided)
    # ----------------------------------

    algo_colors = {
        "smart": "#1f77b4",
        "crisp": "#b41fad",
        "warnless": "#ff7f0e",
        "exida": "#2ca02c",
        "atlas": "#999494"
    }

    # ----------------------------------
    # Plot one figure per metric
        # ----------------------------------

    for metric in ["coverage", "entropy"]:
        metric_df = df[df["Metric"] == metric]

        suts = list(metric_df["SUT"].unique())
        n_suts = len(suts)

        fig, axes = plt.subplots(1, n_suts, figsize=(5 * n_suts, 5), sharey=True)
        if n_suts == 1:
            axes = [axes]

        for ax, sut in zip(axes, suts):
            sut_df = metric_df[metric_df["SUT"] == sut]

            sns.boxplot(
                data=sut_df,
                x="Algorithm",
                y="Value",
                palette=algo_colors,
                ax=ax,
                width=0.6,
            )

            ax.set_title(convert_name(sut), fontsize=17, weight="bold")
            ax.set_ylabel(metric.replace("_", " ").capitalize(), fontsize=14)
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)

            # Style: gray background, white grid, no borders
            ax.set_facecolor("0.8")
            ax.grid(visible=True, which="both", color="white", linestyle="-", linewidth=0.7)
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout()

        save_path = os.path.join(output_path, f"{metric}_boxplot.pdf")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, format="pdf")
        print(f"Boxplots saved to: {save_path}")
        plt.close()
        
    for metric in [
                "avg_max_distance",
                "avg_distance",
                "coverage",
                "entropy",
            ]:
        statistics = {}
        for sut in suts:
            algo_names = list(result[sut].keys())
            values = [result[sut][algo][metric] for algo in algo_names]
            statistics[sut] = AnalysisTables.statistics(values, algo_names)
        data_dict = defaultdict(list)
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                data_dict["Algorithm 1"].append(algorithms[i])
                data_dict["Algorithm 2"].append(algorithms[j])
                for sut in suts:
                    stats = statistics[sut][algorithms[i]][algorithms[j]]
                    if len(stats) > 0:
                        data_dict[f"{sut.capitalize()}.P-Value"].append(stats[0])
                        data_dict[f"{sut.capitalize()}.Effect Size"].append(stats[1])
                    else:
                        data_dict[f"{sut.capitalize()}.P-Value"].append(None)
                        data_dict[f"{sut.capitalize()}.Effect Size"].append(None)
        df = pd.DataFrame(data_dict)
        df = df.dropna()
        df.to_csv(os.path.join(output_path, f"{metric}_stats.csv"), index=False)        
    return result