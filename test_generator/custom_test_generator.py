import random

from model import JudgeResponse, SystemResponse, TestCase
from oracle import Oracle
from sut import SUT

from .test_generator import TestGenerator


class CustomTestGenerator(TestGenerator):
    name: str = "advanced"

    def __init__(
        self,
        manuals: list[str],
        warnings: list[Warning],
        oracle: Oracle,
        sut: SUT,
        **kwargs
    ):
        # TODO your implementation
        pass

    def generate_test(self) -> TestCase:
        # TODO: your implementation
        pass

    def update_state(
        self,
        test: TestCase,
        judge_response: JudgeResponse,
        system_response: SystemResponse,
    ) -> None:
        # TODO: your implementation
        pass