from datetime import datetime, timezone
from typing import List

from wandb import Run
from analysis.llm_figures import boxplots, diversity_report, last_values_table, plot_boxplots_by_algorithm_raw, statistics_table


project = "competition"
metric = "failures"
suts = ["real--gpt-4o-mini","mock--gpt-4o-mini"]
manuals = ["initial", "mini"]

for sut in suts:
    for manual in manuals:
        sut_manual = f"{sut}--{manual}"
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

                if take and run.state == "finished" and \
                    sut in run.name and (manual + "--7200") in run.name:
                    # Time-based filtering
                    if after and not (created > after):
                        continue
                    # if "1" not in run.name or ("2" not in run.name): # just one seed
                    #     continue
                        
                    res.append(run)
            print(f"Filtered runs: {[run.name for run in res]}")
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

        last_values_table(project, ["critical_ratio","failures", "num_warnings_violated"], 
                            run_filters=run_filters,
                            path=f"./wandb_analysis/{sut_manual}/last_values.csv",
                            one_per_name=one_per_name,
                            experiments_folder=experiments_folder,
                            )
        statistics_table(algorithms, 
                        project, "failures", run_filters=run_filters,
                            path=f"./wandb_analysis/{sut_manual}/statistics.csv",
                            one_per_name=one_per_name,
                            experiments_folder=experiments_folder)
        for metric in ["failures", "critical_ratio", "num_warnings_violated"]:
            plot_boxplots_by_algorithm_raw(project, metric=metric, run_filters=run_filters,
                                save_path=f"./wandb_analysis/{sut_manual}/{metric}_boxplots.png",
                                one_per_name=one_per_name,
                                experiments_folder=experiments_folder)
        diversity_report(algorithms,
                        "competition", 
                        input=True, 
                        local_root=experiments_folder,
                        output_path=f"./wandb_analysis/{sut_manual}/diversity",
                        run_filters=run_filters,
                        mode="merged",
                        one_per_name=one_per_name,
                        num_seeds=10)
        # boxplots(algorithms,
        #          project, 
        #         (18, 4), "failures", run_filters=run_filters,
        #         file_name=f"./wandb_analysis/{sut_manual}/failures_boxplots_other.png",
        #         one_per_name=one_per_name,
        #         experiments_folder=experiments_folder)
