from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from typing import Any, Dict, Callable, List

class ChatModelManager:
    def __init__(self):
        self.model_map: Dict[str, Callable[..., Any]] = {
            "claude-3-5-sonnet-20241022": self._create_anthropic_claude_sonnet_model,
            "mistral": self._create_mistral_model,
        }

        self.model_require_args_map: Dict[str, Dict[str, str]] = {
            "claude-3-5-sonnet-20241022": {"api_key": ""},
            "mistral": {"base_url": "localhost:11434"}
        }

    def _create_anthropic_claude_sonnet_model(self, **kwargs) -> ChatAnthropic:
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", **kwargs)

    def _create_mistral_model(self, **kwargs) -> Any:
        return ChatOllama(model="mistral", **kwargs)

    def get_model_list(self) -> List[str]:
        return list(self.model_map.keys())
    
    def get_model_require_args(self, model_name: str) -> Dict[str, str]:
        return self.model_require_args_map[model_name]

    def get_model(self, model_name: str, **kwargs) -> Any:
        if model_name not in self.model_map:
            raise ValueError(f"Model '{model_name}' is not supported. Supported models: {list(self.model_map.keys())}")
        return self.model_map[model_name](**kwargs)