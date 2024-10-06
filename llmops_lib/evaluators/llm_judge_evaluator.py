import json
from llmops_lib.evaluator import Evaluator
from typing import Any, Dict

from langchain_core.runnables.base import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

JUDGE_SYSTEM_PROMPT = """당신은 LLM의 출력 결과를 평가하는 전문가입니다. 주어진 출력(output)과 기준 출력(reference output)을 비교하여, 정확성과 관련성을 기준으로 평가합니다.

평가 기준:
1. **정확성**: 출력이 기준 출력과 동일한 의미 또는 정보를 전달하는가?
2. **관련성**: 출력이 기준 출력의 의도와 목적에 부합하는가?

평가 결과:
- 결과는 이진 점수로 표시:
  - 1: 출력이 기준 출력과 일치하거나 평가 기준을 충족함.
  - 0: 출력이 기준 출력과 일치하지 않거나 평가 기준을 충족하지 못함.

결과는 아래 JSON 형식을 따릅니다.
###
{
    "score": "0 또는 1", 
    "explanation": "점수에 대한 간단한 설명"
}
###
"""

JUDGE_USER_PROMPT = """출력: {output}
기준 출력: {reference_output}
"""

class LLMJudgeEvaluator(Evaluator):
    def __init__(self, chain: Runnable, judge_model: BaseChatModel):
        super().__init__(chain)
        self.judge_model = judge_model

    def evaluate(self, input_variables: dict, reference_output: str) -> Dict[str, Any]:
        result = super().evaluate(input_variables, reference_output)
        llm_judge_result = self.llm_judge(result["output"], reference_output)
        try:
            llm_judge_result = json.loads(llm_judge_result)
        except json.JSONDecodeError:
            llm_judge_result = {"score": -1, "explanation": "LLM judge 결과를 JSON으로 변환하는 중 오류가 발생했습니다."}
        try:
            result["score"] = float(llm_judge_result["score"])
        except ValueError:
            result["score"] = -1.0
            result["scroe_explanation"] = "LLM judge 결과의 점수가 유효한 숫자가 아닙니다."
        result["scroe_explanation"] = llm_judge_result["explanation"]
        return result

    def llm_judge(self, output: str, reference_output: str):
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(JUDGE_SYSTEM_PROMPT, template_format="jinja2"),
            HumanMessagePromptTemplate.from_template(JUDGE_USER_PROMPT, template_format="jinja2")
        ])
        judge_chain = prompt_template | self.judge_model | StrOutputParser()
        response = judge_chain.invoke({
            "output": output,
            "reference_output": reference_output
        })
        return response