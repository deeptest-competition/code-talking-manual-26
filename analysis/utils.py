import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from typing import List, Tuple
from collections import defaultdict
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from analysis.statistics.wilcoxon import run_wilcoxon_and_delaney
import logging
import sys
import math
import tqdm
import matplotlib.pyplot as plt
import random
from collections import Counter
from joblib import Parallel, delayed
import torch

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def close_loggers() -> None:
    # Remove all handlers associated with the root logger object. Needed to call logging.basicConfig multiple times
    # such that for different experiment runs a new log file is written
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


class Log:
    def __init__(self, logger_prefix: str, static_logger: bool = True) -> None:
        self.logger = logging.getLogger(logger_prefix)
        # avoid creating another logger if it already exists
        if len(self.logger.handlers) == 0 or not static_logger:
            self.logger = logging.getLogger(logger_prefix)
            self.logger.setLevel(level=logging.INFO)
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level=logging.DEBUG)

            self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)




class AnalysisException(Exception):
    pass


class AnalysisPlots:
    plots_on_ax = defaultdict(int)
    label_colors = dict()

    @staticmethod
    def interpolate(x: np.ndarray, target_len: int, noise: bool = False):
        x = x.reshape((-1))
        x = [v for v in x]
        while len(x) < target_len:
            new_value = 2 * x[-1] - x[-2]
            if noise:
                new_value *= (random.random() * 0.04 + 0.98)
            x.append(new_value)
        return np.array(x)
   
    @classmethod
    def plot_with_std(cls, ax, dfs: List[pd.DataFrame], label, metric: str = 'failures', color=None, target_time=None, **kwargs):
        """
        Plot mean ± std of multiple runs on the same Axes.
        Parameters:
            ax: matplotlib Axes
            dfs: list of DataFrames (run histories) to plot
            metric: column name to plot
            color: optional, fixed color to use
            kwargs: additional keyword args passed to ax.plot (e.g., label)
        """
        # Initialize counter for this Axes
        if ax not in cls.plots_on_ax:
            cls.plots_on_ax[ax] = 0

        # Use fixed color if provided, else auto
        if color is None:
            if label in cls.label_colors:
                color = cls.label_colors[label]
            else:
                color = f"C{cls.plots_on_ax[ax]}"
                cls.label_colors[label] = color
            cls.plots_on_ax[ax] += 1

        # Prepare data
        xs = [df[metric].to_numpy() for df in dfs]
        if target_time is not None:
            xs = [AnalysisPlots.interpolate(x, target_time+1) for x in xs]
        min_len = min(target_time + 1, min(x.shape[0] for x in xs))
        xs = np.concatenate([x[:min_len].reshape(-1, 1) for x in xs], axis=1)
        avg = xs.mean(1)
        std = np.std(xs, 1)

        # Convert time index to minutes
        if target_time is None:
            time = dfs[0].index[:min_len].total_seconds() / 60  # assuming TimedeltaIndex
        else:
            time = np.arange(target_time + 1)
        # Plot mean
        ax.plot(time, avg, color=color, **kwargs)

        # Fill ± std
        ax.fill_between(x=time, y1=avg-std, y2=avg+std, color=color, alpha=0.5)

    @classmethod
    def boxplot(cls, ax, dfs_lists: List[List[pd.DataFrame]], labels, metric: str = 'failures',**kwargs):
        # Prepare data
        xs = [[df[metric].iloc[-1] for df in dfs] for dfs in dfs_lists]
        # Plot 
        bplot = ax.boxplot(xs, patch_artist=True, **kwargs)
        for patch, label in zip(bplot['boxes'], labels):
            if label in cls.label_colors:
                color = cls.label_colors[label]
            else:
                color = f"C{len(cls.label_colors)}"
                cls.label_colors[label] = color
            patch.set_facecolor(color)

    @staticmethod
    def plot_clusters(
        to_cluster,
        centers,
        filename,
        names,
        env_configurations_names,
        visualize_name=None,
        seed=0,
    ):
        colors = [
            "black",
            "green",
            "blue",
            "purple",
            "magenta",
            "pink",
            "grey",
            "cyan",
            "lime",
            "orange",
            "peru",
        ]
        markers = ["o", "^", "2", "*", "s", "+", "x", "D", "h", "8", ">"]
        for c in centers:
            # add center of cluster as a new row
            to_cluster = np.append(to_cluster, c.reshape(1, -1), axis=0)

        for per in [2, 30]:
            # assuming TSNE
            n_iterations = 5000
            tsne = TSNE(
                n_components=2,
                perplexity=per,
                n_jobs=-1,
                n_iter=n_iterations,
                random_state=seed,
            )
            embeddings = tsne.fit_transform(to_cluster)

            _ = plt.figure()
            ax = plt.gca()
            ax.tick_params(left=False)
            ax.tick_params(bottom=False)
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticklabels([])

            count_configs = 0
            name_idx = 0

            embeddings_names = dict()
            for i in range(len(embeddings) - len(centers)):
                if (
                    count_configs
                    > env_configurations_names[names[name_idx]] - 1
                ):
                    count_configs = 0
                    name_idx += 1

                if names[name_idx] not in embeddings_names:
                    embeddings_names[names[name_idx]] = ([], [])
                embeddings_names[names[name_idx]][0].append(embeddings[i][0])
                embeddings_names[names[name_idx]][1].append(embeddings[i][1])

                count_configs += 1

            for nm, ems in embeddings_names.items():
                assert len(ems[0]) == env_configurations_names[nm], "Number of embeddings {} != Number of failures: {} for {}".format(
                    len(ems[0]), env_configurations_names[nm], nm
                )

            for i, name_embeddings in enumerate(embeddings_names.items()):
                nm, embeddings_ = name_embeddings

                if visualize_name is None or visualize_name == nm:
                    plt.scatter(
                        embeddings_[0],
                        embeddings_[1],
                        s=80,
                        color=colors[i],
                        marker=markers[i],
                        label=nm,
                        alpha=0.6
                    )

            embeddings_centroids = ([], [])
            for i in range(
                len(embeddings) - 1, len(embeddings) - 1 - len(centers), -1
            ):
                embeddings_centroids[0].append(embeddings[i][0])
                embeddings_centroids[1].append(embeddings[i][1])

            plt.scatter(
                embeddings_centroids[0],
                embeddings_centroids[1],
                s=50,
                color="red",
                marker="$c$",
                label="centroid",
            )

            plt.legend(prop={"size": 6})

            plt.savefig(f"{filename}_per{per}.pdf", format="pdf")
            plt.clf()
            plt.close()


class AnalysisTables:
    @staticmethod
    def statistics(values: List[List], algo_names: List[str]):
        result = defaultdict(lambda: defaultdict(tuple))
        for i in range(len(algo_names)):
            for j in range(i + 1, len(algo_names)):
                algo1 = algo_names[i]
                algo2 = algo_names[j]
                data1 = values[i]
                data2 = values[j]
                if len(data1) == len(data2) and len(data1) > 2:
                    stats = run_wilcoxon_and_delaney(data1, data2)
                    result[algo1][algo2] = stats
                    stats = run_wilcoxon_and_delaney(data2, data1)
                    result[algo2][algo1] = stats
        return result


class AnalysisDiversity:
    @staticmethod
    def average_max_distance(embeddings: np.ndarray):
        distances = distance_matrix(embeddings, embeddings)
        return distances.max(0).mean()
    
    @staticmethod
    def average_distance(embeddings: np.ndarray):
        distances = distance_matrix(embeddings, embeddings)
        return distances.mean(0).mean()    

    @staticmethod
    def cluster_data(
        data: np.ndarray,
        n_clusters_interval: tuple,
        seed: int,
        silhouette_threshold: int,
        n_jobs: int = -1,
    ):
        logger = Log("cluster_data")

        min_k, max_k = n_clusters_interval
        optimal_n_clusters = 1
        optimal_score = -1.0

        def evaluate_k(n_clusters):
            print("Applying clustering for k = ",n_clusters)
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init="auto",
            )
            labels = clusterer.fit_predict(data)
            score = silhouette_score(data, labels)
            return n_clusters, score

        if min_k != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_k)(k)
                for k in tqdm.tqdm(range(min_k, max_k))
            )

            for n_clusters, silhouette_avg in results:
                if silhouette_avg > optimal_score:
                    if silhouette_threshold < 0 or optimal_score < 0:
                        optimal_score = silhouette_avg
                        optimal_n_clusters = n_clusters
                    else:
                        percentage_increase = (
                            100 * (silhouette_avg - optimal_score) / optimal_score
                        )
                        if percentage_increase >= silhouette_threshold:
                            optimal_score = silhouette_avg
                            optimal_n_clusters = n_clusters

        clusterer = KMeans(
            n_clusters=optimal_n_clusters,
            random_state=seed,
            n_init="auto",
        ).fit(data)

        return (
            clusterer,
            clusterer.labels_,
            clusterer.cluster_centers_,
            optimal_score,
        )
        
    @staticmethod
    def compute_coverage_entropy(labels, centers, names, env_configurations_names, device='cuda'):
        """
        Compute coverage, entropy, and Gini impurity on GPU using PyTorch.

        Args:
            labels: list or 1D array of cluster labels (integers)
            centers: list of cluster centers
            names: list of names corresponding to segments of labels
            env_configurations_names: dict mapping name -> number of labels
            device: 'cuda' or 'cpu'
        Returns:
            coverage_names: dict of coverage percentage per name
            entropy_names: dict of entropy percentage per name
            num_clusters: total number of clusters
        """

        # ---------------------------
        # Setup
        # ---------------------------
        labels = torch.tensor(labels, device=device, dtype=torch.int64)
        num_clusters = len(centers)
        ideal_entropy = math.log2(num_clusters)
        unique_names = list(env_configurations_names.keys())
        num_classes = len(unique_names)
        
        print(f"Labels tensor device: {labels.device}")

        # Compute slices for each name
        counts_per_name = torch.tensor([env_configurations_names[n] for n in names], device=device)
        name_cuts = torch.cumsum(counts_per_name, dim=0)

        # ---------------------------
        # Prepare storage
        # ---------------------------
        coverage_names = {}
        entropy_names = {}
        gini_dict = {}

        start_idx = 0

        # Wrap the enumerate(names) with tqdm
        for idx, nm in enumerate(tqdm.tqdm(names, desc="Computing coverage & entropy")):
            n_points = env_configurations_names[nm]
            end_idx = start_idx + n_points
            points = labels[start_idx:end_idx]

            # ---------------------------
            # Compute coverage
            # ---------------------------
            unique_clusters = torch.unique(points)
            coverage = 100.0 * unique_clusters.numel() / num_clusters
            coverage_names[nm] = float(coverage)

            # ---------------------------
            # Distribution / counts per cluster
            # ---------------------------
            counts = torch.bincount(points, minlength=num_clusters)
            distribution = 100.0 * counts / n_points

            # ---------------------------
            # Entropy
            # ---------------------------
            nonzero_counts = counts[counts > 0].float()
            freqs = nonzero_counts / n_points
            entropy = -(freqs * torch.log2(freqs)).sum() if nonzero_counts.numel() > 0 else torch.tensor(0.0, device=device)
            entropy_percentage = 100.0 * entropy / ideal_entropy if ideal_entropy > 0 else 0.0
            entropy_names[nm] = float(entropy_percentage)

            # ---------------------------
            # Gini for two-class case
            # ---------------------------
            if num_classes == 2:
                total_counts = torch.bincount(labels, minlength=num_clusters).float()
                probs = torch.zeros(num_clusters, device=device)
                for cluster_label in torch.unique(points):
                    probs[cluster_label] = (points == cluster_label).sum().float() / total_counts[cluster_label]
                gini_dict[nm] = probs

            start_idx = end_idx


                # ---------------------------
                # Compute overall Gini impurity
                # ---------------------------
        if num_classes == 2 and gini_dict:
            # Preallocate list of probabilities for stacking
            probs_list = []

            # Wrap the iteration over names with tqdm
            for nm in tqdm(gini_dict.keys(), desc="Computing Gini probabilities"):
                probs_list.append(gini_dict[nm])

            # Stack into a single tensor
            probs_matrix = torch.stack(probs_list)  # shape: (num_names, num_clusters)

            # Optional sanity check
            if not torch.allclose(probs_matrix.sum(dim=0), torch.tensor(1.0, device=device)):
                raise RuntimeError(f"Error in computing Gini coefficient: {gini_dict}")

            # Compute Gini impurity
            gini_impurity_coeff = (1 - (probs_matrix**2).sum(dim=0)).mean().item()
            gini_purity_coeff = 1 - gini_impurity_coeff

            print(f"Gini impurity coefficient: {gini_impurity_coeff:.4f}")
            print(f"Gini purity coefficient: {gini_purity_coeff:.4f}")

        return coverage_names, entropy_names, num_clusters