from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    groundedness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    reasoning: str


JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator scoring an AI
agent's response for a drug-discovery intelligence task.

Question: {question}
Retrieved context the agent had access to: {context}
Agent's response: {response}

Score groundedness (0-1: does every claim trace back to the context?) and
relevance (0-1: does it actually answer the question?). Be strict about
groundedness - any unsupported factual claim should sharply lower the score."""


class LLMJudge:
    def __init__(self, model_name: str, provider: str = "groq"):
        if provider == "groq":
            from langchain_groq import ChatGroq

            base_llm = ChatGroq(model=model_name, temperature=0.0)
        else:
            from langchain_anthropic import ChatAnthropic

            base_llm = ChatAnthropic(model=model_name, temperature=0.0)
        self.llm = base_llm.with_structured_output(JudgeScore)

    def score(self, question: str, context: str, response: str) -> JudgeScore:
        prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, response=response)
        return self.llm.invoke(prompt)
