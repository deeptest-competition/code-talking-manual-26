from typing import Literal

from pydantic import BaseModel


class Warning(BaseModel):
    id: str
    extra_ids: list[str]
    warning_text: str
    top_section_id: str
    top_section_title: str
    feedback: list[str] = []
    success: int = 1
    failure: int = 1


class TestCase(BaseModel):
    request: str
    expected_warning_id: str
    warning_text: str = ""


class JudgeResponse(BaseModel):
    justification: str = ""
    score: float = 1.0
    status: Literal["ok", "error"] = "ok"


class SystemResponse(BaseModel):
    answer: str
    documents: list


class TestResult(BaseModel):
    test_case: TestCase
    system_response: SystemResponse = None
    judge_response: JudgeResponse = None
    timestamp: float = ""
    is_valid: bool = True
    validity_check_failure_reason: str = ""

class MetaData(BaseModel):
    generator_name: str
    total_time: float
    time_limit_seconds: int
    timestamp: str
    sut: str
    oracle: str
    config: dict
    
class Metrics(BaseModel):
    total_tests: int
    num_failures: int
    average_distance: float
    num_warnings_violated: int