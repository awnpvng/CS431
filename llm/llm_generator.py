from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterator

class LLMType(Enum):
    GEMINI = "gemini"
    CLAUDE = "claude"
    GROQ = "groq"

@dataclass
class LLMConfig:
    llm_type: LLMType
    api_key: str 
    model_name: str = "" #optional

@dataclass
class LLMResponse:
    content: str
    llm_type: LLMType
    model_name: str

class LLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        #thêm thuộc tính logging bằng thư viện logging
    @abstractmethod
    #dùng cho các text output như explanation,...
    def generate_text(self, user_prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        pass
    @abstractmethod
    #dùng cho streaming output với Streamlit st.write_stream()
    def generate_text_stream(self, user_prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        pass


class LLMGenerator:
    def __init__(self, config: LLMConfig):
        self.strategy = self._choose_strategy(config)
        
    def _choose_strategy(self, config: LLMConfig) -> LLMClient:
        from .gemini_client import GeminiClient
        # from .groq_client import GroqClient
        
        strategies = {
            LLMType.GEMINI: GeminiClient,
            # LLMType.GROQ: GroqClient,
            #LLMType.CLAUDE: ClaudeClient,
        }
        strategy_class = strategies.get(config.llm_type)
        if not strategy_class:
            raise ValueError(f"Unsupported LLM type: {config.llm_type}")
        return strategy_class(config)
       
    def generate_text(self, user_prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        return self.strategy.generate_text(user_prompt, system_prompt)
    def generate_text_stream(self, user_prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        return self.strategy.generate_text_stream(user_prompt, system_prompt)

    #bổ sung cải tiến: check config đã hợp lệ hay chưa? 
