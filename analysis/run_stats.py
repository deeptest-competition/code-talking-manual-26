import argparse
from datetime import datetime, timezone
from typing import List

from wandb import Run
from analysis.llm_figures import (
    boxplots,
    diversity_report,
    last_values_table,
    plot_boxplots_by_algorithm_raw,
    statistics_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W&B analysis for different SUT and manual configurations."
    )
    parser.add_argument(
        "--suts",
        nargs="+",
        default=["real--gpt-4o-mini", "mock--gpt-4o-mini"],
        help="List of systems under test (SUTs).",
    )
    parser.add_argument(
        "--manuals",
        nargs="+",
        default=["initial", "mini"],
        help="List of manuals to evaluate.",
    )
    parser.add_argument(
        "--do_coverage",
        action="store_true",
        help="If set, compute diversity/coverage reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project = "competition"
    algorithms = ["crisp", "exida", "smart", "warnless", "atlas"]
    experiments_folder = "wandb_download"
    one_per_name = True

    for sut in args.suts:
        for manual in args.manuals:
            sut_manual = f"{sut}--{manual}"

            def filter_runs(runs: List[Run], after: datetime = None) -> List[Run]:
                res = []
                for run in runs:
                    created = run.created_at
                    if isinstance(created, str):
                        created = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )

                    if (
                        run.state == "finished" 
                        and sut in run.name
                        and (manual + "--7200") in run.name
                    ):
                        if after and not (created > after):
                            continue
                        res.append(run)

                print(f"Filtered runs: {[run.name for run in res]}")
                return res

            run_filters = {"competition": filter_runs}

            # last_values_table(
            #     project,
            #     ["critical_ratio", "failures", "num_warnings_violated"],
            #     run_filters=run_filters,
            #     path=f"./wandb_analysis/{sut_manual}/last_values.csv",
            #     one_per_name=one_per_name,
            #     experiments_folder=experiments_folder,
            # )

            # statistics_table(
            #     algorithms,
            #     project,
            #     "failures",
            #     run_filters=run_filters,
            #     path=f"./wandb_analysis/{sut_manual}/statistics.csv",
            #     one_per_name=one_per_name,
            #     experiments_folder=experiments_folder,
            # )

            for metric in ["failures", "critical_ratio", "num_warnings_violated"]:
                plot_boxplots_by_algorithm_raw(
                    project,
                    metric=metric,
                    run_filters=run_filters,
                    save_path=f"./wandb_analysis/{sut_manual}/{metric}_boxplots.pdf",
                    one_per_name=one_per_name,
                    experiments_folder=experiments_folder,
                )

            if args.do_coverage:
                diversity_report(
                    algorithms,
                    project,
                    input=True,
                    local_root=experiments_folder,
                    output_path=f"./wandb_analysis/{sut_manual}/diversity",
                    run_filters=run_filters,
                    mode="merged",
                    one_per_name=one_per_name,
                    num_seeds=10,
                )

if __name__ == "__main__":
    main()
