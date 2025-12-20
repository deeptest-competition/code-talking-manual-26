from abc import ABC, abstractmethod

from model import SystemResponse, TestCase


class SUT(ABC):
    @abstractmethod
    def ask(self, test_case: TestCase) -> SystemResponse:
        pass
