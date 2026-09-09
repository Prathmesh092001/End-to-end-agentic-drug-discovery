import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from partex_agent.entity import EvaluationConfig
from partex_agent.logging import logger

METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


class RagasEvaluator:
    """
    Runs the RAGAS suite over a benchmark of (question, ground_truth,
    retrieved_contexts, generated_answer) rows. This is the automated
    gate that runs in CI before a prompt/config change ships - if
    faithfulness drops below the configured threshold, the pipeline fails.
    """

    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg
        self.metrics = [METRIC_MAP[m] for m in cfg.ragas_metrics if m in METRIC_MAP]

    def evaluate_dataframe(self, df: pd.DataFrame) -> dict:
        required_cols = {"question", "answer", "contexts", "ground_truth"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Benchmark dataframe missing columns: {missing}")

        dataset = Dataset.from_pandas(df)
        result = evaluate(dataset, metrics=self.metrics)
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        logger.info(f"RAGAS scores: {scores}")

        faithfulness_score = scores.get("faithfulness", 1.0)
        passed = faithfulness_score >= self.cfg.groundedness_threshold
        if not passed:
            logger.warning(
                f"faithfulness {faithfulness_score} below threshold {self.cfg.groundedness_threshold}"
            )
        return {"scores": scores, "passed": passed}
