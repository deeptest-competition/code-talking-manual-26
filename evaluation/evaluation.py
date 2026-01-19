import numpy as np
from scipy.spatial import distance_matrix
import wandb

from llm.utils.embeddings_local import get_embedding
from model import Metrics, TestResult, MetaData
from llm.llms import ModelStatistics
from utils.console import print_evaluation_summary
from .export import *


def evaluate(results: list[TestResult]) -> Metrics:
    failures = [r for r in results if r.judge_response is not None and r.judge_response.score < 0.5]
    num_failures = len(failures)
    if num_failures == 0:
        num_failures, average_distance, num_warnings_violated, failure_ratio = 0, 0, 0, 0
    else:
        embeddings = np.array([get_embedding(r.test_case.request) for r in failures])
        distances = distance_matrix(embeddings, embeddings)
        average_distance = np.mean(distances)
        warning_ids = set(r.test_case.expected_warning_id for r in failures)
        num_warnings_violated = len(warning_ids)
        failure_ratio = num_failures / len(results)
    return Metrics(
        total_tests=len(results),
        num_failures=num_failures,
        average_distance=average_distance,
        num_warnings_violated=num_warnings_violated,
        failure_ratio=failure_ratio,
    )
    
def save_all(
    results: list[TestResult],
    metadata: MetaData,
    save_folder: str,
):
    metrics = evaluate(results)
    
    save_llm_usage_statistics(save_folder)
    save_results(results=results, save_folder=save_folder)
    save_evaluation_results(metrics=metrics, save_folder=save_folder)
    save_metadata(metadata=metadata, save_folder=save_folder)
    
    print_evaluation_summary(metrics.dict(),
                             save_folder,
                             other_params={
                                "total_time (s):" : round(metadata.total_time, 3)
                                })
    

def wandb_log(results: list[TestResult],
              metadata: MetaData,
              save_folder: str,
              wandb_entity: str,
              wandb_project: str,
              args: dict):
    run_name = "_".join([
        args['sut_type'],
        args['oracle_type'],
        args['sut_llm'] if args['sut_llm'] else "defaultSUTLLM",
        args['oracle_llm'] if args['oracle_llm'] else "defaultOracleLLM",
        args['test_generator'],
        args['manual_name'],
        str(args['n_tests']) + "tests",
        str(args['seed']),
    ])
    with wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        tags=[f"{k}:{str(v)[:32]}" for k, v in args.items()],
    ) as run:
        metrics = evaluate(results)
        run.log(metrics.model_dump())
        run.log({"total_time (s)": round(metadata.total_time, 3)})
        usage_summary = ModelStatistics.complete_statistics()
        usage_summary["total_tokens"] = ModelStatistics.total_values()
        run.log(usage_summary)
        artifact = wandb.Artifact(
            name=run_name + "_results",
            type="output_directory",
        )
        artifact.add_dir(save_folder)
        run.log_artifact(artifact)
        