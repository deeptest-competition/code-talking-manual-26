from abc import ABC, abstractmethod
from model import TestCase


class Validator(ABC):
    @abstractmethod
    def validate(self, test_case: TestCase) -> [bool,str]:
        pass
