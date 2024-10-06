import time
import os
from typing import Any, Dict, Callable
from enum import Enum
from langchain_core.runnables.base import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

class EvaluatorType(Enum):
    EXACT_MATCH = "ExactMatchEvaluator"
    EMBEDDING_DISTANCE = "EmbeddingDistanceEvaluator"
    LLM_JUDGE = "LLMJudgeEvaluator"
    RAG = "RAGEvaluator"

class Evaluator:
    def __init__(self, chain: Runnable):
        self.chain = chain

    def evaluate(self, input_variables: dict, reference_output: str) -> Dict[str, Any]:
        start_time = time.time()
        response = self.chain.invoke(input_variables)
        latency = time.time() - start_time

        evaluation_result = {
            "input_variables": input_variables,
            "output": response.content,
            "reference_output": reference_output,
            "input_token": response.usage_metadata["input_tokens"],
            "output_token": response.usage_metadata["output_tokens"],
            "latency": latency,
        }

        return evaluation_result

def create_evaluator(
        evaluator_type: EvaluatorType, 
        chain: Runnable, 
        judge_model: BaseChatModel = None,
        embedding_model: Embeddings = None,
        retriever: BaseRetriever = None,
    ) -> Evaluator:
    if evaluator_type == EvaluatorType.EXACT_MATCH:
        from llmops_lib.evaluators import ExactMatchEvaluator
        return ExactMatchEvaluator(chain)
    elif evaluator_type == EvaluatorType.EMBEDDING_DISTANCE:
        if embedding_model is None:
            raise ValueError("embedding_model function must be provided for EmbeddingDistanceEvaluator")
        from llmops_lib.evaluators import EmbeddingDistanceEvaluator
        return EmbeddingDistanceEvaluator(chain=chain, embedding_model=embedding_model)
    elif evaluator_type == EvaluatorType.LLM_JUDGE:
        if judge_model is None:
            raise ValueError("judge_model function must be provided for LLMJudgeEvaluator")
        from llmops_lib.evaluators import LLMJudgeEvaluator
        return LLMJudgeEvaluator(chain=chain, judge_model=judge_model)
    elif evaluator_type == EvaluatorType.RAG:
        if retriever is None:
            raise ValueError("retriever function must be provided for RAGEvaluator")
        if judge_model is None:
            raise ValueError("judge_model function must be provided for RAGEvaluator")
        from llmops_lib.evaluators import RAGEvaluator
        return RAGEvaluator(chain=chain, retriever=retriever, evaluator_llm=judge_model)
    else:
        raise ValueError(f"Unsupported evaluator type: {evaluator_type}")