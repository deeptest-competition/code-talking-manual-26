import json
from pathlib import Path
import pickle
from typing import Dict, List, Literal, Optional, Tuple

import tqdm

from wandb import Run
from opensbt.utils.wandb import download_run_artifacts, download_run_artifacts_relative, get_summary
from opensbt.visualization.utils import AnalysisDiversity, AnalysisPlots, AnalysisException, AnalysisTables
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tqdm
from llm.model.models import Utterance
from llm.utils.embeddings_local import get_embedding as get_embedding_local
from llm.utils.embeddings_openai import get_embedding as get_embedding_openai

algo_names_map = {
    ("gs", "astral"): "ASTRAL",
    ("gs", "extended"): "T-wise",
    "gs": "T-wise",
    "rs": "Random",
    "nsga2": "STELLAR",
    "NSGA2" : "STELLAR",
    "nsga2d": "NSGAIID",
}

metric_names = {
    "failures": "Number of Failures",
    "critical_ratio": "Critical Ratio",
}

def convert_name(run_name: str) -> str:
    """
    Convert a run name into a standardized SUT string by scanning each
    word (separated by '_') in order and including all matching identifiers.
    """
    sut_keywords = {
        "ipa": "",
        "yelp": "",
        "gpt-4o": "GPT-4o",
        "gpt-5-chat": "GPT-5-Chat",
        "deepseek-v3-0324": "DeepSeek-V3",
        "chatbmw": "",
        "mistral": "Mistral-7B",
        "qwen3" : "Qwen3-8B",
        "deepseek-v2" : "DeepSeek-V2-16B"
    }

    words = run_name.split("_")
    sut_parts = []
    for w in words:
        w_lower = w.lower()
        if w_lower in sut_keywords:
            print(w_lower)
            kwd = sut_keywords[w_lower]
            if kwd != "":
                sut_parts.append(kwd)
    
    return "_".join(sut_parts) if sut_parts else "unknown_sut"

def get_algo_name(algo, features):
    algo_name = algo_names_map.get((algo, features), None)
    if algo_name is None:
        algo_name = algo_names_map[algo]
    return algo_name

def capitalize(name: str) -> str:
    return "_".join(word.capitalize() for word in name.split("_"))

def get_embeddings(artifact_directory_path: str, critical_only: bool = True,
                   local: bool = True, input: bool = True) -> Tuple[List[Utterance], np.ndarray]:
    file = "embeddings.pkl" if input else "embeddings_output.pkl"
    pickle_path = os.path.join(artifact_directory_path, file)
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
        
    get_embedding = get_embedding_local if local else get_embedding_openai
    print(f"Calculating embeddings for {artifact_directory_path}")
    json_path = "all_critical_utterances.json" if critical_only else "all_utterances.json"
    with open(os.path.join(artifact_directory_path, json_path), "r", encoding="utf8") as f:
        data = json.load(f)
    utterances = []
    embeddings = []
    for obj in data:
        utterance = obj.get("utterance", {})
        fitness = obj.get("fitness", {})
        
        question = utterance.get("question", "")
        answer = utterance.get("answer", "")
        if answer is None:
            answer = ""
        if question is None:
            question = ""
        answer = answer.strip()
        question = question.strip()
        is_critical = obj.get("is_critical", None)
        content_fitness = fitness.get("content_fitness", None)
        if question != "":
            if not critical_only:
                utterances.append(utterance)
                embeddings.append(
                    get_embedding(question).reshape(1, -1) if input else get_embedding(answer).reshape(1, -1))
                continue
            if is_critical and (content_fitness is None or content_fitness < 1.0):
                utterances.append(utterance)
                embeddings.append(
                    get_embedding(question).reshape(1, -1) if input else get_embedding(answer).reshape(1, -1))
                               
    embeddings = np.concatenate(embeddings)
    with open(pickle_path, "wb") as f:
        pickle.dump((utterances, embeddings), f)
    return utterances, embeddings


def get_summary(artifact_directory_path: str):
    json_path = os.path.join(artifact_directory_path, "evaluation_summary.json")
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary

def get_real_tests(path_name:str, th_content: float = None, th_response: float = None) -> Tuple[int,int]:
    all_tests = os.path.join(path_name, "all_utterances.json")

    # Initialize counters
    num_real_tests = 0
    num_real_fail = 0

    # Load the JSON data
    with open(all_tests, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure data is iterable (e.g., a list of objects)
    if isinstance(data, dict):
        data = [data]

    num_crit_dupfree = int(get_summary(path_name)["Number Critical Scenarios (duplicate free)"])
    num_crit = int(get_summary(path_name)["Number Critical Scenarios"])

    num_all_dupfree = int(get_summary(path_name)["Number All Scenarios (duplicate free)"])
    num_all = int(get_summary(path_name)["Number All Scenarios"])

    # Iterate through each object
    for obj in data:
        utterance = obj.get("utterance", {})
        fitness = obj.get("fitness", {})
        
        question = utterance.get("question", "").strip()
        is_critical = obj.get("is_critical", None)
        content_fitness = fitness.get("content_fitness", None)
        answer_fitness = fitness.get("answer_fitness", None)

        # Check for real test (non-empty question)
        if question:
            num_real_tests += 1

            # Increment also if content_fitness is None or < 1.0
            if is_critical and (content_fitness is None or content_fitness < 1.0):

                # Determine failure conditions
                no_thresholds = (th_content is None and th_response is None)
                content_fail = (th_content is not None and content_fitness is not None and content_fitness <= th_content)
                response_fail = (th_response is not None and answer_fitness is not None and answer_fitness <= th_response)

                # Increment fail count if no thresholds defined OR any threshold is violated
                if no_thresholds or content_fail or response_fail:
                    num_real_fail += 1
    
    num_real_fail = num_real_fail - (num_crit - num_crit_dupfree)
    num_real_tests = num_real_tests - (num_all - num_all_dupfree)

    # print(f"num_real_tests: {num_real_tests}")
    # print(f"num_real_fail: {num_real_fail}")
    return num_real_tests, num_real_fail



def plot_metric_vs_time(
    project="SafeLLM",
    size=(16, 6),
    metric="failures",
    time_in_minutes=180,
    file_name="plot",
    run_filters = None,
    one_per_name: bool = False,
    experiments_folder: str = rf"C:\Users\levia\Documents\testing\LLM\opensbt-llm\wandb_download",
    tight: bool = False 
):  
    print("Project name:", project)
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts_relative(f"opentest/{project}", 
                                                     local_root=experiments_folder, 
                                                     filter_runs=run_filters[project],
                                                     one_per_name=one_per_name)

    print("len(artifact_paths):", len(artifact_paths))
    fig, axes = plt.subplots(1, len(artifact_paths), sharey="row", sharex="all")

    # Flatten axes to a 1D list for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    # Fixed color map for algorithms
    algo_colors = {
        "STELLAR": "tab:green",
        "NSGAII": "tab:green",
        "T-wise": "tab:orange",
        "Random": "tab:blue",
        "ASTRAL": "tab:red"
    }
    
    # add as needed
    fig.set_size_inches(*size)
    fig.supxlabel("Time [min]")
    # fig.supylabel(metric_names[metric])
    tick_count = time_in_minutes // 30 + 1
    ticks_kwargs = {"xticks": np.linspace(0, time_in_minutes, tick_count)}
    if metric == "critical_ratio":
        ticks_kwargs["yticks"] = np.linspace(0.0, 1.0, 6)
        ticks_kwargs["ylim"] = (0.0, 1.0)
    plt.setp(axes, **ticks_kwargs)
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)

            print("algo_name:", algo_name)
            color = algo_colors.get(algo_name, None)

            dfs = [get_run_history_table(path) for path in paths]
            AnalysisPlots.plot_with_std(axes[i], 
                                        dfs,
                                        label=algo_name,
                                        metric=metric,
                                        color = color,
                                        target_time=time_in_minutes)
        axes[i].set_title(convert_name(sut))
        axes[i].set_box_aspect(1)
        axes[i].tick_params(axis="x", rotation=45)  # rotate x-axis labels
    
    axes[0].set_ylabel(metric_names[metric], labelpad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    if tight:
        plt.tight_layout()

    # Set figure background to light gray

    # Set axes background to gray as well
    for ax in axes:
        ax.set_facecolor("0.93")  # slightly darker than figure background
        ax.grid(
            visible=True,
            which="both",
            color="white",  # light grid lines
            linestyle="-",
            linewidth=0.8
        )
           # Remove all borders (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.savefig(file_name)

def get_color_by_algo(algo):
    print("algo", algo)
    algo = algo.lower()

    if algo == "rs":
        return "tab:blue"
    elif algo == "gs":
        return "tab:orange"
    elif algo == "nsga2":
        return "tab:green"
    else:
        return "black"
    
def plot_metric_vs_time_ivan(
    project="SafeLLM",
    size=(18, 6),
    metric="failures",
    time_in_minutes=120,
    file_name="plot",
    run_filters=None,
    experiments_folder = "wandb_experiments",
    one_per_name=False,
    plot_legend = False,
    tight=False,
    th_content=None,
    th_response=None
):
    
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts_relative(f"opentest/{project}", 
                                                     local_root=experiments_folder,
                                                     filter_runs = run_filters[project],
                                                     one_per_name=one_per_name)
    fig, axes = plt.subplots(1, len(artifact_paths), sharey="row", sharex="all")
    fig.set_size_inches(*size)

    # Ensure axes is always a list
    if len(artifact_paths) == 1:
        axes = [axes]
    # fig.supylabel(metric_names[metric])
    tick_count = time_in_minutes // 30 + 1
    ticks_kwargs = {"xticks": np.linspace(0, time_in_minutes, tick_count)}
    if metric == "critical_ratio":
        ticks_kwargs["yticks"] = np.linspace(0.0, 1.0, 6)
        ticks_kwargs["ylim"] = (0.0, 1.0)
    plt.setp(axes, **ticks_kwargs)
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            color = get_color_by_algo(algo)
            dfs = [get_run_history_table(path, th_response=th_response, th_content=th_content) for path in paths]
            AnalysisPlots.plot_with_std(axes[i], dfs, label=algo_name, metric=metric, target_time=time_in_minutes, color = color)
        axes[i].set_title(convert_name(sut.capitalize()))
        axes[i].set_box_aspect(1)
        axes[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.85) 
    
    # Put y-label only next to first subplot
    axes[0].set_ylabel(metric_names[metric], labelpad=10, fontsize=20)

    if len(axes) > 1:
        fig.supxlabel("Time [min]", fontsize = 20)
    else:
        axes[0].set_xlabel("Time [min]", fontsize = 20)

    if plot_legend:
        legend_handles = {}
        for label, col in AnalysisPlots.label_colors.items():
            legend_handles[label] = plt.Line2D([0], [0], color=col, lw=10)
        fig.legend(legend_handles.values(), legend_handles.keys(), title="Labels", loc="upper right")
    if tight:
        plt.tight_layout()
    
    #fig.subplots_adjust(wspace=0.05)  # reduce horizontal space between subplots

    folder = os.path.dirname(file_name)
    
    os.makedirs(folder, exist_ok=True)
    plt.savefig(file_name)

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

    num_tests_all, num_real_fail = get_real_tests(path_name = run_path, th_content = th_content, th_response = th_response)
    
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
        axes[i].set_title(convert_name(sut.capitalize()))
        axes[i].set_box_aspect(1)
        axes[i].set_xticklabels(algo_names, rotation='vertical')

    # Label y-axis for the first subplot only
    axes[0].set_ylabel(metric_names[metric], labelpad=10, fontsize=20)

    legend_handles = {
        label: plt.Line2D([0], [0], color=col, lw=10)
        for label, col in AnalysisPlots.label_colors.items()
    }

    fig.tight_layout()
    plt.savefig(file_name)

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
            elif metric == "critical_ratio":
                for path in paths:
                    num_real_tests, num_real_fail = get_real_tests(path)
                    ratio = num_real_fail / num_real_tests if num_real_tests > 0 else 0.0
                    values.append(ratio)
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
        "STELLAR": "#1f77b4",
        "NSGAII": "#1f77b4",
        "Random": "#ff7f0e",
        "T-wise": "#2ca02c",
    }

    suts = list(df["SUT"].unique())
    n_suts = len(suts)

    # Define the desired full order
    full_order = ["STELLAR", "T-wise", "Random"]

    # # Keep only the algorithms present in the dataset
    # algo_order = [algo for algo in full_order if algo in df["Algorithm"].unique()]
    algo_order = full_order

    # Ensure the column is categorical with this order
    df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=algo_order, ordered=True)

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
            order=algo_order
        )

        ax.set_title(convert_name(sut), fontsize=23, weight="bold")  # title = model name
        ax.set_ylabel(metric.replace("_", " ").capitalize(), fontsize = 20)
        ax.set_xlabel("")  # remove x-axis label
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        # Style: gray background, white grid, no borders
        ax.set_facecolor("0.9")
        ax.grid(visible=True, which="both", color="white", linestyle="-", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
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
                elif metric == "critical_ratio":
                    values = []
                    for path in paths:
                        num_real_tests, num_real_fail = get_real_tests(path)
                        ratio = num_real_fail / num_real_tests if num_real_tests > 0 else 0.0
                        values.append(ratio)
                else:
                    raise AnalysisException(f"Unknown metric: {metric}")

                row[f"{metric}_mean"] = np.mean(values) if values else np.nan
                row[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0

            summary_data.append(row)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    algorithm_order = ["STELLAR", "Random", "T-wise"]
    summary_df = pd.DataFrame(summary_data)
    summary_df["Algorithm"] = pd.Categorical(summary_df["Algorithm"], categories=algorithm_order, ordered=True)
    summary_df = summary_df.sort_values(by=["SUT", "Algorithm"]).reset_index(drop=True)

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
        df.to_csv(os.path.join(output_path, f"{metric}_stats.csv"), index=False)        
    return result

# def diversity_report(
#         algorithms,
#         project="SafeLLM",
#         output_path="diversity",
#         input: bool = True,
#         max_num_clusters = 150,
#         silhouette_threshold = 20,
#         visualize: bool = False,
#         local_root: str = None,
#         run_filters: dict = None,
#         mode: Literal["separated", "merged"] = "merged",
#         num_seeds: Optional[int] = 10,
#         one_per_name: bool = True,
#         save_folder: str = ""
#     ):
#     output_path = os.path.join(output_path, "input" if input else "output")

#     if local_root != None:
#         artifact_paths = download_run_artifacts_relative(f"opentest/{project}", 
#                                                          local_root=local_root,
#                                                          filter_runs = run_filters[project],
#                                                          one_per_name=one_per_name)
#     else:
#         artifact_paths = download_run_artifacts(f"opentest/{project}", run_filters[project])
    
#     print(f"Applying diversity analysis for {len(artifact_paths)} runs.")
#     cached_embeddings = dict()
#     suts = []
#     result = {}
#     if os.path.exists(os.path.join(output_path, "report.json")):
#         with open(os.path.join(output_path, "report.json"), "r") as f:
#             result = json.load(f)
#     else:
#         for i, (sut, algos) in enumerate(artifact_paths.items()):
#             suts.append(sut)
#             print(sut)
#             result[sut] = dict()
#             algo_names = [get_algo_name(*key) for key in algos.keys()]

#             for algo_name, paths in zip(algo_names, algos.values()):
#                 avg_max_distances = []
#                 avg_distances = []
#                 for path in paths:
#                     _, embeddings = get_embeddings(path)
#                     cached_embeddings[(path, input)] = embeddings
#                     avg_max_distances.append(AnalysisDiversity.average_max_distance(embeddings))
#                     avg_distances.append(AnalysisDiversity.average_distance(embeddings))
#                 result[sut][algo_name] = {
#                     "avg_max_distance": avg_max_distances,
#                     "avg_distance": avg_distances,
#                 }
#             for nm in algo_names:
#                 result[sut][nm]["coverage"] = []
#                 result[sut][nm]["entropy"] = []

#             if mode == "separated":
#                 original_seeds = min([len(paths) for paths in algos.values()])
#                 if num_seeds is None:
#                     num_seeds = original_seeds
#                 for seed in range(num_seeds):
#                     algo_counts = dict()
#                     to_cluster = []
#                     for algo_name, paths in zip(algo_names, algos.values()):
#                         embeddings = cached_embeddings[(paths[seed % original_seeds], input)]
#                         algo_counts[algo_name] = embeddings.shape[0]
#                         to_cluster.append(embeddings)
#                     to_cluster = np.concatenate(to_cluster)
#                     max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
#                     clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
#                         data=to_cluster,
#                         n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
#                         seed=seed,
#                         silhouette_threshold=silhouette_threshold,
#                     )
#                     coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
#                         labels, centers, algo_names, algo_counts,
#                     )
#                     for nm in coverage:
#                         result[sut][nm]["coverage"].append(coverage[nm])
#                         result[sut][nm]["entropy"].append(entropy[nm])
#                     os.makedirs(os.path.join(output_path, sut), exist_ok=True)
#                     cluster_data = (
#                         to_cluster,
#                         algo_names,
#                         algo_counts,
#                         labels,
#                         centers,
#                         seed,
#                     )
#                     with open(pickled_data_path, "wb") as f:
#                         pickle.dump(cluster_data, f)
#                     if visualize:
#                         AnalysisPlots.plot_clusters(
#                             to_cluster,
#                             centers,
#                             os.path.join(output_path, sut, f"clusters_seed{seed}"),
#                             algo_names,
#                             algo_counts,
#                             seed=seed,
#                         )
#             elif mode == "merged":
#                 algo_counts = defaultdict(int)
#                 to_cluster = []
#                 for algo_name, paths in tqdm.tqdm(zip(algo_names, algos.values())):
#                     for path in paths:
#                         embeddings = cached_embeddings[(path, input)]
#                         algo_counts[algo_name] += embeddings.shape[0]
#                         to_cluster.append(embeddings)
#                 to_cluster = np.concatenate(to_cluster)
#                 max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
#                 for seed in range(num_seeds):
#                     pickled_data_path = os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl")
                    
                    
#                     if os.path.exists(pickled_data_path):
#                         with open(pickled_data_path, "rb") as f:
#                             content = pickle.load(f)
#                             (
#                                 to_cluster,
#                                 algo_names,
#                                 algo_counts,
#                                 labels,
#                                 centers,
#                                 seed,
#                             ) = content       
#                     else:
#                         clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
#                             data=to_cluster,
#                             n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
#                             seed=seed,
#                             silhouette_threshold=silhouette_threshold,
#                         )
#                         os.makedirs(os.path.join(output_path, sut), exist_ok=True)   
#                         cluster_data = (
#                             to_cluster,
#                             algo_names,
#                             algo_counts,
#                             labels,
#                             centers,
#                             seed,
#                         )
#                         with open(os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl"), "wb") as f:
#                             pickle.dump(cluster_data, f)

#                     coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
#                         labels, centers, algo_names, algo_counts,
#                     )
#                     for nm in coverage:
#                         result[sut][nm]["coverage"].append(coverage[nm])
#                         result[sut][nm]["entropy"].append(entropy[nm])                   
#                     if visualize:
#                         AnalysisPlots.plot_clusters(
#                             to_cluster,
#                             centers,
#                             os.path.join(output_path, sut, f"clusters_seed{seed}"),
#                             algo_names,
#                             algo_counts,
#                             seed=seed,
#                         )
#             else:
#                 raise AnalysisException("Unknown mode")
#             print(result)

#         with open(os.path.join(output_path, "report.json"), "w") as f:
#             json.dump(result, f)

#     data_dict = defaultdict(list)
#     suts = list(artifact_paths.keys())
#     for algorithm in algorithms:
#         data_dict["Algorithm"].append(algorithm)
#         for sut in suts:
#             for metric in [
#                 "avg_max_distance",
#                 "avg_distance",
#                 "coverage",
#                 "entropy",
#             ]:
#                 if algorithm not in result[sut]:
#                     mean = None
#                 else:
#                     values = result[sut][algorithm].get(metric, None)
#                     mean = np.mean(values) if values is not None else None
#                 data_dict[f"{sut}.{metric}"].append(mean)
#     pd.DataFrame(data_dict).to_csv(os.path.join(output_path, "scores.csv"), index=False)

    
#     for metric in [
#                 "avg_max_distance",
#                 "avg_distance",
#                 "coverage",
#                 "entropy",
#             ]:
#         statistics = {}
#         for sut in suts:
#             algo_names = list(result[sut].keys())
#             values = [result[sut][algo][metric] for algo in algo_names]
#             statistics[sut] = AnalysisTables.statistics(values, algo_names)
#         data_dict = defaultdict(list)
#         for i in range(len(algorithms)):
#             for j in range(i + 1, len(algorithms)):
#                 data_dict["Algorithm 1"].append(algorithms[i])
#                 data_dict["Algorithm 2"].append(algorithms[j])
#                 for sut in suts:
#                     stats = statistics[sut][algorithms[i]][algorithms[j]]
#                     if len(stats) > 0:
#                         data_dict[f"{sut.capitalize()}.P-Value"].append(stats[0])
#                         data_dict[f"{sut.capitalize()}.Effect Size"].append(stats[1])
#                     else:
#                         data_dict[f"{sut.capitalize()}.P-Value"].append(None)
#                         data_dict[f"{sut.capitalize()}.Effect Size"].append(None)
#         df = pd.DataFrame(data_dict)
#         df = df.dropna()
#         df.to_csv(os.path.join(output_path, save_folder, f"{metric}_stats.csv"), index=False)        
#     return result
