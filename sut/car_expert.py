import requests

from model import SystemResponse, TestCase

from .sut import SUT
from config import get_config
config = get_config()


class CarExpertAdapter(SUT):
    def __init__(self, url: str | None = None):
        self.url = url or config["sut"]["car_expert_url"]

    def ask(self, test_case: TestCase) -> SystemResponse:
        headers = {
            "content-type": "application/json",
        }
        payload = {
            "history": {
                "history": [],
            },
        }
        request_url = f"{self.url}/answer?question={test_case.request}"
        response = requests.post(request_url, json=payload, headers=headers)
        if response.status_code != 200:
            return SystemResponse(
                answer=f"Error: Received status code {response.status_code} ({response.content}) from Car Expert service.",
                documents=[],
            )
        data = response.json()
        answer = data.get("result", "No answer provided.")
        documents = data.get("source_documents", [])
        return SystemResponse(answer=answer, documents=documents)
