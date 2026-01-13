import argparse
import json

from llm.llms import LLMType
from oracle import SimpleJudge
from pipeline import Pipeline
from sut import MockCarExpert
from test_generator import (CustomTestGenerator, SimpleTestGenerator,
                            SmartTestGenerator)
from utils.manual import load_manuals_from_directory
from utils.retriever import Retriever
from utils.warnings import read_warnings_from_csv
from config import get_config

import warnings as warnings_filter

warnings_filter.filterwarnings(
    "ignore",
    message=".*encoder_attention_mask.*",
    category=FutureWarning,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Car Expert Warning Tests")
    parser.add_argument(
        "--time_limit_seconds",
        type=int,
        default=10,
        help="Time limit for the evaluation in seconds (default: 10)",
    )
    parser.add_argument(
        "--n_tests",
        type=int,
        default=None,
        help="Number of tests to generate (default: None, meaning unlimited)",
    )
    parser.add_argument(
        "--manual_path",
        type=str,
        default="./data",
        help="Path to the directory containing manual files (default: ./data)",
    )
    parser.add_argument(
        "--warnings_csv",
        type=str,
        default="./data/warnings.csv",
        help="Path to the CSV file containing warnings (default: ./data/warnings.csv)",
    )
    parser.add_argument(
        "--test_generator",
        type=str,
        default="smart",
        help="Type of test generator to use (default: smart)",
    )
    parser.add_argument(
        "--sut_type",
        type=str,
        default="mock",
        help="Type of System Under Test (SUT) to use (default: mock)",
    )
    parser.add_argument(
        "--oracle_type",
        type=str,
        default="simple",
        help="Type of oracle to use (default: simple)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=rf"./configs/default_config.json",
        help=rf"Path to the overall config file.",
    )
    parser.add_argument(
        "--sut_llm",
        type=str,
        default="gpt-4o",
        help="Type of LLM to use for the SUT (default: gpt-4o)",
    )
    parser.add_argument(
        "--oracle_llm",
        type=str,
        default="gpt-4o",
        help="Type of LLM to use for the oracle (default: gpt-4o)",
    )
    parser.add_argument(
        "--generator_llm",
        type=str,
        default="gpt-4o-mini",
        help="Type of LLM to use for the generation. Optional (default: gpt-4o).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generation (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    documents = load_manuals_from_directory(args.manual_path)
    warnings = read_warnings_from_csv(args.warnings_csv)
    retriever = Retriever(args.manual_path)

    if args.sut_type == "mock":
        sut = MockCarExpert(
            retriever, llm_type=LLMType(args.sut_llm) if args.sut_llm else None
        )
    elif args.sut_type == "real":
        raise NotImplementedError("Real SUT not provided during development")
    else:
        raise ValueError(f"Unknown SUT type: {args.sut_type}")

    if args.oracle_type == "simple":
        oracle = SimpleJudge(
            retriever, llm_type=LLMType(args.oracle_llm) if args.oracle_llm else None
        )
    elif args.oracle_type == "real":
        raise NotImplementedError("Real oracle not provided during development")
    else:
        raise ValueError(f"Unknown oracle type: {args.oracle_type}")
    
    print("[Oracle] Using LLM:", LLMType(args.oracle_llm))
    
    if args.test_generator == "smart":
        generator_type = SmartTestGenerator
    elif args.test_generator == "simple":
        generator_type = SimpleTestGenerator
    elif args.test_generator == "custom":
        generator_type = CustomTestGenerator
    elif args.test_generator == "warnless":
        from test_generator.warnless_test_generator import WarnlessTestGenerator
        generator_type = WarnlessTestGenerator
    elif args.test_generator == "atlas":
        from test_generator.atlas_test_generator import AtlasTestGenerator
        generator_type = AtlasTestGenerator
    elif args.test_generator == "exida":
        from test_generator.exida_test_generator import ExidaTestGenerator
        generator_type = ExidaTestGenerator
    else:
        raise ValueError(f"Unknown test generator type: {args.test_generator}")

    config = get_config(args.config_path)
    generator_config = config.get("generator", {})

    if args.generator_llm and "llm_type" in generator_config:
        generator_config["llm_type"] = args.generator_llm

    results = Pipeline.evaluate_generator(
        oracle=oracle,
        sut=sut,
        generator_type=generator_type,
        generator_kwargs=generator_config,
        documents=documents,
        warnings=warnings,
        num_tests=args.n_tests,
        time_limit_seconds=args.time_limit_seconds,
        seed=args.seed
    )
