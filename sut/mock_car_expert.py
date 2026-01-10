from llm.llms import LLMType, pass_llm
from model import SystemResponse, TestCase
from utils.retriever import Retriever

from .sut import SUT
from config import get_config
config = get_config()

class MockCarExpert(SUT):
    def __init__(self, retriever: Retriever, llm_type: LLMType = None):
        self.retriever = retriever
        self.llm_type = llm_type or LLMType[config["sut"]["llm_type"]]

        print("[MockCarExpert] Using LLM: ", self.llm_type)

    def ask(self, test_case: TestCase) -> SystemResponse:
        docs = self.retriever.retrieve_info(test_case.request)
        context = "\n\n".join([d["content"] for d in docs])
        prompt = f"""
        You are a concise in-car assistant summarizing the vehicle manual.
        Use only the context below to answer the question accurately.
        Mention all warnings if relevant.

        Guidelines:
        - Answer concisely in 1–3 sentences.
        - Do not hallucinate information not present in the context.
        - If the answer is not in the context, respond apporpriately without guessing.
        - Treat warnings with high importance. If relevant, include them in your answer.
        - Any missed warning can have sever consequences to the user.

        Context:
        {context}

        Question:
        {test_case.request}

        Answer briefly in 1–3 sentences. Do not hallucinate.
        """
        answer = pass_llm(msg=prompt, 
                          llm_type=self.llm_type,
                          )
        return SystemResponse(answer=answer, documents=docs)
