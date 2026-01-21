from datetime import datetime, timezone
from typing import List

from wandb import Run
from analysis.llm_figures import diversity_report, last_values_table, plot_boxplots_by_algorithm_raw, statistics_table


project = "competition"
metric = "failures"

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


run_filters = {"competition": filter_runs}

algorithms = [
    "crisp",
    "exida",
    "smart",
    "warnless",
    "atlas"
]


experiments_folder = rf"wandb_download"
one_per_name = True

path = rf"./wandb_analysis/test"

diversity_report(algorithms,
                 "competition", 
                 input=True, 
                 local_root=experiments_folder,
                 output_path=path,
                 run_filters=run_filters,
                 mode="merged",
                 one_per_name=one_per_name,
                 num_seeds=10)