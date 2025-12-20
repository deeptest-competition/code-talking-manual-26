import time
from copy import deepcopy
from typing import Type

from evaluation.evaluation import save_all
from evaluation.export import create_save_folder
from model import TestResult, Warning, MetaData, SystemResponse, JudgeResponse
from oracle import Oracle
from sut import SUT
from test_generator import TestGenerator
import datetime
import zoneinfo
from validator.simple_validator import SimpleValidator
from utils.console import print_error, print_user, print_sut, print_judge
from config import get_config

config = get_config()
            
import random
class Pipeline:
    @staticmethod
    def evaluate_generator(
        oracle: Oracle,
        sut: SUT,
        generator_type: Type[TestGenerator],
        generator_kwargs: dict,
        documents: list[str],
        warnings: list[Warning],
        num_tests: int | None = None,
        time_limit_seconds: int | None = None,
    ) -> None:
        save_folder = create_save_folder(generator_type.name, config["results"]["output_path"])
        generator: TestGenerator = generator_type(
            documents, warnings, deepcopy(oracle), deepcopy(sut), **generator_kwargs
        )
        results: list[TestResult] = []

        start_time = time.time()
        total_time = 0

        validator = SimpleValidator()

        while True:
            if num_tests is not None and len(results) >= num_tests:
                break
            total_time = (time.time() - start_time) 
            if (
                time_limit_seconds is not None
                and total_time > time_limit_seconds
            ):
                break

            test = generator.generate_test()

            print_user(test.request)

            is_valid, reason = validator.validate(test)

            # if the test is not valid skip execution and evaluation
            if not is_valid:
                print_error(f"Generated test input not valid. {reason}")
                results.append(TestResult(
                        test_case=test,
                        timestamp=time.time(),
                        is_valid = False,
                        validity_check_failure_reason = reason
                    )
                )
                is_valid = True
                continue
            
            system_response = sut.ask(test)
            print_sut(system_response.answer)

            judge_response = oracle.evaluate(test, system_response)
            print_judge(judge_response.justification)

            generator.update_state(test, judge_response, system_response)

            results.append(
                TestResult(
                    test_case=test,
                    system_response=system_response,
                    judge_response=judge_response,
                    timestamp=time.time(),
                )
            )

        tz = zoneinfo.ZoneInfo("Europe/Berlin")
        dt_local = datetime.datetime.fromtimestamp(start_time, tz)
        timestamp = dt_local.isoformat()
        
        metadata = MetaData(generator_name = generator.name,
                           timestamp = timestamp,
                           time_limit_seconds = time_limit_seconds,
                           total_time = total_time,
                           sut = sut.__class__.__name__,
                           oracle = oracle.__class__.__name__,
                           config = config
                           )
        save_all(results=results, 
                 metadata=metadata, 
                 save_folder=save_folder)
