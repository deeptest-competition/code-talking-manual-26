import json
import os
from datetime import datetime
from pathlib import Path

from llm.llms import ModelStatistics
from model import Metrics, TestResult


def save_results(results: list[TestResult], save_folder: str) -> str:
    """
    Serializes a list of TestResult instances into a timestamped JSON file.
    Returns the path to the written file.
    """
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    file_path = save_folder + os.sep + f"evaluated_tests.json"

    # Pydantic models provide .model_dump() in v2 (or .dict() in v1)
    payload = [r.model_dump() for r in results]

    with Path(file_path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return file_path


def save_llm_usage_statistics(save_folder: str):
    with open(save_folder + "llm_usage_summary.json", "w") as f:
        usage_summary = ModelStatistics.complete_statistics()
        usage_summary["total_tokens"] = ModelStatistics.total_values()
        json.dump(usage_summary, f, indent=4)


def create_save_folder(name, base_folder: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"{base_folder}/{name}_{timestamp}/"
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    return save_folder


def save_evaluation_results(
    metrics: Metrics,
    save_folder: str,
):
    with open(save_folder + "evaluation_summary.json", "w") as f:
        json.dump(metrics.model_dump(), f, indent=4)

def save_metadata(metadata, save_folder):    
    with open(save_folder + "metadata.json", "w") as f:
        json.dump(metadata.model_dump(), f, indent=4)