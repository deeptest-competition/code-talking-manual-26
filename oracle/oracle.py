from abc import ABC, abstractmethod

from model import JudgeResponse, SystemResponse, TestCase, Warning


class Oracle(ABC):
    @abstractmethod
    def evaluate(
        self, test_case: TestCase, system_response: SystemResponse
    ) -> JudgeResponse:
        pass
