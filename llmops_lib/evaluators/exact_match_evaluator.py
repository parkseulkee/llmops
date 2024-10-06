from typing import Any, Dict
from llmops_lib.evaluator import Evaluator

class ExactMatchEvaluator(Evaluator):
    def evaluate(self, input_variables: dict, reference_output: str) -> Dict[str, Any]:
        result = super().evaluate(input_variables, reference_output)
        result["score"] = float(result["output"] == reference_output)
        return result