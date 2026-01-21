import io
import time
import os
import json
from typing import Callable, Optional, List, Dict, Tuple
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import tqdm

import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.wandb_run import Run
from wandb.apis.public import Api
import re

def filter_runs(runs: List[Run], after: datetime = None) -> List[Run]:
    res = []
    take = True
    for run in runs:
        # print("run name:", run.name)
        # Ensure created_at is timezone-aware
        created = run.created_at
        if isinstance(created, str):
            # Parse ISO 8601 string from W&B into datetime
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))

        if take and run.state == "finished":
            # Time-based filtering
            if after and not (created > after):
                continue
                
            res.append(run)

    return res

def get_sut(project: str, run, tags: dict) -> str: 
    sut_name = "_".join([tags["sut_type"], tags["sut_llm"], tags["manual_name"]])
    return sut_name

def download_run_artifacts_relative(
    wb_project_path: str,
    local_root: str = "./wandb_competition",
    filter_runs: Optional[Callable[[List[Run]], List[Run]]] = None,
    one_per_name: bool = False
) -> Dict[str, str]:
    """
    Download W&B run artifacts.
    
    Parameters
    ----------
    wb_project_path : str
        The "entity/project" path in W&B.
    local_root : str, default="./opentest"
        Local root directory for downloaded artifacts.
    filter_runs : callable, optional
        Function to filter runs before download.
    one_per_name : bool, default=False
        If True, only one run (latest) per unique run.name is downloaded.
        If False, all runs are downloaded.
    """
    api = Api()
    runs = api.runs(wb_project_path, per_page=1000)
    project_name = wb_project_path.split("/")[-1]

    if filter_runs is not None:
        runs = filter_runs(runs)

    runs = list(runs)

    print(f"Found {len(runs)} runs before grouping.")

    # Optionally group by run.name
    if one_per_name:
        grouped = defaultdict(list)
        for run in runs:
            grouped[run.name].append(run)
        selected_runs = [max(group, key=lambda r: r.created_at) for group in grouped.values()]
        print(f"Reduced to {len(selected_runs)} unique run names.")
    else:
        selected_runs = runs
        print(f"Downloading all {len(selected_runs)} runs.")

    all_paths = defaultdict(lambda: defaultdict(list))
    
    for run in tqdm.tqdm(selected_runs):
        tags = {t.split(":")[0]: t.split(":")[1] for t in run.tags}
        sut_name = get_sut(project_name, run, tags)

        local_path = os.path.join(
            local_root,
            project_name,
            "artifacts",
            sut_name,
            tags.get("test_generator", "unknown_algo"),
            tags.get("seed", "no_seed"),
            run.name,
            run.id
        )
        os.makedirs(local_path, exist_ok=True)

        all_paths[sut_name][(tags.get("test_generator", "unknown_algo"), 
                             tags.get("test_generator", "default"))].append(local_path)

        # Skip if directory already contains files
        if any(os.scandir(local_path)):
            continue

        # Download all logged artifacts
        for artifact in run.logged_artifacts():
            artifact.download(local_path)

        # Save tags metadata
        with open(os.path.join(local_path, "tags.json"), "w") as f:
            json.dump(tags, f)

        # Save full run history
        history_df = run.history(keys=None, pandas=True)
        history_df.to_csv(os.path.join(local_path, "run_history.csv"), index=False)

    print(f"Downloaded {len(all_paths)} run directories.")
    print(f"all_paths:", all_paths)
    return all_paths

if __name__ == "__main__":
    download_run_artifacts_relative(
        wb_project_path="opentest/competition", 
        filter_runs=filter_runs,
        local_root="./wandb_competition", one_per_name=True)