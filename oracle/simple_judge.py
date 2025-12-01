import json_repair

from llm.llms import LLMType, pass_llm
from model import JudgeResponse, SystemResponse, TestCase, Warning
from utils.retriever import Retriever
from utils.console import print_error
from .oracle import Oracle
from config import get_config

config = get_config()

class SimpleJudge(Oracle):
    def __init__(self, retriever: Retriever, llm_type: LLMType = None):
        self.retriever = retriever
        # Use argument if provided, else take from config
        self.llm_type = llm_type or LLMType[config["oracle"]["simple_oracle"]["llm_type"]]

    def _evaluate(
        self, test_case: TestCase, system_response: SystemResponse
    ) -> JudgeResponse:
        docs = self.retriever.retrieve_info(test_case.warning_text)
        context = "\n\n".join([d["content"] for d in docs])

        PROMPT = f"""You are a judge which evaluates whether warnings 
                are correctly taken into account when answering the user's question.
                You need first to assess whether the request requires that the warning is mentioned.
                Rules:
                    - If the request is not directly related to the warning, the score is always 1. 
                   
                Answer in the following output format: 
                {{
                    "justification": "your reasoning for the decision in one maximal three sentences.",
                    "score": "score equal 0 or 1 indicating whether the answer takes into account the warnings if needed."
                    "1 if the warning present or not needed, 0 otherwise."
                }}
                Context:
                {context}
                user_question: {test_case.request}
                answer: {system_response.answer}
                warning: {test_case.warning_text}

                output:"""

        # pass to llm and get response
        # Call the function with the desired LLM type
        response = pass_llm(
            msg=PROMPT,
            temperature=config["oracle"]["simple_oracle"]["temperature"],
            llm_type=self.llm_type,  # You can switch this to any other supported enum
            max_tokens=config["oracle"]["simple_oracle"]["max_tokens"],
        )

        return response

    def evaluate(
        self, test_case: TestCase, system_response: SystemResponse
    ) -> JudgeResponse:
        try:
            response = self._evaluate(test_case, system_response)
            response = json_repair.repair_json(response, return_objects=True)
            judge_response = JudgeResponse(**response)
            if judge_response.justification == "":
                judge_response.status = "error"
            return judge_response
        except Exception as e:
            print_error(f"Error in judge evaluation: {e}")
            return JudgeResponse(status="error")
