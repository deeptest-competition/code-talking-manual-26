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
from llm.utils.embeddings_local import get_batch_embeddings
from llm.utils.embeddings_local import get_embedding as get_embedding_local
from llm.utils.embeddings_openai import get_embedding as get_embedding_openai
from analysis.models import Utterance
from tqdm import tqdm
from analysis.config import manual_warnings
from analysis.llm_figures import *

all_paths ={'real--gpt-4o-mini--mini': {('exida', 'exida'): ['wandb_download/competition/artifacts/real--gpt-4o-mini--mini/exida/1/real--gpt-4o-mini--simple--gpt-4o-mini--exida--mini--7200--1/2aodybfx', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/exida/2/real--gpt-4o-mini--simple--gpt-4o-mini--exida--mini--7200--2/6s494akv', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/exida/3/real--gpt-4o-mini--simple--gpt-4o-mini--exida--mini--7200--3/8lkqx2j4', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/exida/4/real--gpt-4o-mini--simple--gpt-4o-mini--exida--mini--7200--4/tyqp2bq4', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/exida/5/real--gpt-4o-mini--simple--gpt-4o-mini--exida--mini--7200--5/g14h6x3j', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/exida/6/real--gpt-4o-mini--simple--gpt-4o-mini--exida--mini--7200--6/3ixcd077'], ('warnless', 'warnless'): ['wandb_download/competition/artifacts/real--gpt-4o-mini--mini/warnless/1/real--gpt-4o-mini--simple--gpt-4o-mini--warnless--mini--7200--1/yyyil0ha', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/warnless/2/real--gpt-4o-mini--simple--gpt-4o-mini--warnless--mini--7200--2/sl2jtks2', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/warnless/3/real--gpt-4o-mini--simple--gpt-4o-mini--warnless--mini--7200--3/80gus6hr', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/warnless/4/real--gpt-4o-mini--simple--gpt-4o-mini--warnless--mini--7200--4/qipiekzw', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/warnless/5/real--gpt-4o-mini--simple--gpt-4o-mini--warnless--mini--7200--5/6wucr69o', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/warnless/6/real--gpt-4o-mini--simple--gpt-4o-mini--warnless--mini--7200--6/dun3dtyr'], ('smart', 'smart'): ['wandb_download/competition/artifacts/real--gpt-4o-mini--mini/smart/1/real--gpt-4o-mini--simple--gpt-4o-mini--smart--mini--7200--1/ds6tya2a', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/smart/2/real--gpt-4o-mini--simple--gpt-4o-mini--smart--mini--7200--2/k3e07fyr', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/smart/3/real--gpt-4o-mini--simple--gpt-4o-mini--smart--mini--7200--3/un7xqcgk', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/smart/4/real--gpt-4o-mini--simple--gpt-4o-mini--smart--mini--7200--4/dofj1ftu', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/smart/5/real--gpt-4o-mini--simple--gpt-4o-mini--smart--mini--7200--5/ry262l5a', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/smart/6/real--gpt-4o-mini--simple--gpt-4o-mini--smart--mini--7200--6/qtmmo4jx'], ('crisp', 'crisp'): ['wandb_download/competition/artifacts/real--gpt-4o-mini--mini/crisp/1/real--gpt-4o-mini--simple--gpt-4o-mini--crisp--mini--7200--1/g5bmuuzp', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/crisp/2/real--gpt-4o-mini--simple--gpt-4o-mini--crisp--mini--7200--2/3vslamdj', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/crisp/3/real--gpt-4o-mini--simple--gpt-4o-mini--crisp--mini--7200--3/rcdsbq4h', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/crisp/4/real--gpt-4o-mini--simple--gpt-4o-mini--crisp--mini--7200--4/b91pymix', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/crisp/5/real--gpt-4o-mini--simple--gpt-4o-mini--crisp--mini--7200--5/cy3zh7c2', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/crisp/6/real--gpt-4o-mini--simple--gpt-4o-mini--crisp--mini--7200--6/q3n0avho'], ('atlas', 'atlas'): ['wandb_download/competition/artifacts/real--gpt-4o-mini--mini/atlas/1/real--gpt-4o-mini--simple--gpt-4o-mini--atlas--mini--7200--1/9xxmjkxc', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/atlas/2/real--gpt-4o-mini--simple--gpt-4o-mini--atlas--mini--7200--2/pa9ixwcz', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/atlas/3/real--gpt-4o-mini--simple--gpt-4o-mini--atlas--mini--7200--3/vj7ap4nd', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/atlas/4/real--gpt-4o-mini--simple--gpt-4o-mini--atlas--mini--7200--4/m34ipho2', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/atlas/5/real--gpt-4o-mini--simple--gpt-4o-mini--atlas--mini--7200--5/wu5i7efe', 'wandb_download/competition/artifacts/real--gpt-4o-mini--mini/atlas/6/real--gpt-4o-mini--simple--gpt-4o-mini--atlas--mini--7200--6/dklkuxhv']}}


def plot_boxplots_by_algorithm_raw(
    project: str = "competition",
    metric: str = "failures",
    run_filters=None,
    one_per_name: bool = False,
    experiments_folder: str = "wandb_download",
    save_path: str = "plots/boxplots_raw.pdf"
):
    """
    Create boxplots of raw run values for each algorithm (one subplot per algorithm).
    Uses the same artifact data structure as in last_values_table().
    """

    artifact_paths = all_paths

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
    full_order = ["atlas", "crisp", "exida", "smart", "warnless"]

    # Keep only the algorithms present in the dataset
    algo_order = [algo for algo in full_order if algo in df["Algorithm"].unique()]
    algo_order = full_order

    df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=algo_order, ordered=True)

    # Create one subplot per SUT
    fig, axes = plt.subplots(1, n_suts, figsize=(5 * n_suts, 5), sharey=True)

    if n_suts == 1:
        axes = [axes]

    for ax, sut in zip(axes, suts):
        sut_df = df[df["SUT"] == sut]

        algo_map = {
            "crisp": "CRISP",
            "exida": "Exida",
            "atlas" : "ATLAS",
            "smart" : "Random",
            "warnless" : "Warnless"
            # add others as needed
        }

        sns.boxplot(
            data=sut_df,
            x="Algorithm",
            y="Value",
            palette=algo_colors,
            ax=ax,
            width=0.6,
        )

        # ax.set_xticklabels(
        #     [algo_map.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()]
        # )

        ax.set_xticklabels(
            [algo_map.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
            rotation=45,   # steeper angle
            ha="right",
            fontsize=18    # smaller font
        )
        # Optional: fine-tune horizontal position
        # Shift labels to the right in axis coordinates
     
        # ax.set_title(convert_name(sut), fontsize=16)  # title = model name
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=18)
        ax.set_xlabel("")  # remove x-axis label
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=18)
        # Style: gray background, white grid, no borders
        ax.set_facecolor("0.9")
        ax.grid(visible=True, which="both", color="white", linestyle="-", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, format="pdf")
    print(f"Boxplots saved to: {save_path}")
    plt.close()
    
    
# ------------------ Example run ------------------
if __name__ == "__main__":
    plot_boxplots_by_algorithm_raw(metric="failures", save_path="plots/boxplots_raw.pdf")