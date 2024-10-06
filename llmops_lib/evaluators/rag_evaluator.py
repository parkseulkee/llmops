from typing import Any, Dict, List
from llmops_lib.evaluator import Evaluator

from langchain_core.runnables.base import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import Document
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, Faithfulness

class RAGEvaluator(Evaluator):
    def __init__(self, chain: Runnable, retriever: BaseRetriever, evaluator_llm: BaseChatModel):
        super().__init__(chain)
        self.retriever = retriever # 검색기
        self.evaluator_llm = LangchainLLMWrapper(evaluator_llm) # Ragas 평가자 모델

        # 평가 지표 정의
        self.context_precision = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
        self.faithfulness = Faithfulness(llm=self.evaluator_llm)

    def evaluate(self, input_variables: dict, reference_output: str) -> Dict[str, Any]:
        result = super().evaluate(input_variables, reference_output)
        
        # 입력 질문
        question = input_variables.get("question", "")
        if question is None:
            raise ValueError("The question cannot be None.")
        # 검색된 문서
        retrieved_documents = self.retriever.invoke(question)

        # 싱글턴 평가
        sample = SingleTurnSample(
            user_input=input_variables["question"],
            response=result["output"],
            retrieved_contexts=[doc.page_content for doc in retrieved_documents]
        )

        # 평가 지표 계산
        context_precision_score = self.context_precision.single_turn_score(sample)
        faithfulness_score = self.faithfulness.single_turn_score(sample)
        
        # 결과에 지표 포함
        result["context_precision"] = context_precision_score
        result["faithfulness"] = faithfulness_score
        
        # 평가 지표의 평균을 최종 점수로 사용
        result["score"] = (context_precision_score + faithfulness_score) / 2

        return result