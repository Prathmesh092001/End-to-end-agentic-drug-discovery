from pydantic import BaseModel

from partex_agent.entity import AgentConfig


class BaseAgent:
    """
    Common LLM setup for every specialist agent. Keeps provider-switching
    (Groq / Anthropic / OpenAI) and structured-output binding in one place
    so each agent file only contains its prompt and domain logic.

    Defaults to Groq because it has a genuinely free tier - swap
    `llm.provider` in config.yaml to "anthropic" or "openai" once you
    have paid keys, with zero changes needed here or in any agent file.
    """

    def __init__(self, cfg: AgentConfig, output_schema: type[BaseModel] | None = None):
        self.cfg = cfg
        provider = cfg.llm.provider

        if provider == "groq":
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=cfg.llm.model_name,
                temperature=cfg.temperature,
                max_tokens=cfg.llm.max_tokens,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                model=cfg.llm.model_name,
                temperature=cfg.temperature,
                max_tokens=cfg.llm.max_tokens,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=cfg.llm.model_name,
                temperature=cfg.temperature,
                max_tokens=cfg.llm.max_tokens,
            )
        else:
            raise ValueError(f"Unknown llm provider '{provider}'. Use groq | anthropic | openai.")

        self.llm = llm.with_structured_output(output_schema) if output_schema else llm
