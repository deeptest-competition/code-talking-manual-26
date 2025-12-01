import random
from abc import ABC, abstractmethod

from model import JudgeResponse, SystemResponse, TestCase, Warning
from oracle import Oracle
from sut import SUT


class TestGenerator(ABC):
    def __init__(
        self,
        manuals: list[str],
        warnings: list[Warning],
        oracle: Oracle,
        sut: SUT,
        **kwargs
    ):
        self.manuals = manuals
        self.warnings = warnings
        self.oracle = oracle
        self.sut = sut

    @abstractmethod
    def generate_test(self) -> TestCase:
        pass

    @abstractmethod
    def update_state(
        self,
        test: TestCase,
        judge_response: JudgeResponse,
        system_response: SystemResponse,
    ) -> None:
        pass
