from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from partex_agent.entity import EvaluationConfig
from partex_agent.logging import logger

METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


class RagasEvaluator:
    """Runs RAGAS evaluation over benchmark data (question, ground_truth,

    contexts, answer). Serves as an automated quality gate in CI/CD.
    """

    def __init__(
        self,
        cfg: EvaluationConfig,
        evaluator_llm: Optional[Any] = None,
        evaluator_embeddings: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.metrics = [METRIC_MAP[m] for m in cfg.ragas_metrics if m in METRIC_MAP]
        self.evaluator_llm = evaluator_llm
        self.evaluator_embeddings = evaluator_embeddings

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates columns and ensures 'contexts' is always list[str] for RAGAS."""
        required_cols = {"question", "answer", "contexts", "ground_truth"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Benchmark dataframe missing columns: {missing}")

        formatted_df = df.copy()

        # Fix contexts column to ensure it is list[str] across all rows
        def _normalize_context(val):
            if isinstance(val, list):
                return [str(item) for item in val if item is not None]
            if pd.isna(val) or val is None:
                return ["No retrieval context provided."]
            return [str(val)]

        formatted_df["contexts"] = formatted_df["contexts"].apply(_normalize_context)
        formatted_df["ground_truth"] = formatted_df["ground_truth"].apply(
            lambda g: [str(g)] if isinstance(g, str) else list(g)
        )

        return formatted_df

    def evaluate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates pipeline outputs against ground truth with RAGAS metrics."""
        formatted_df = self._sanitize_dataframe(df)
        dataset = Dataset.from_pandas(formatted_df)

        logger.info(f"Evaluating {len(formatted_df)} queries with RAGAS...")

        # Run evaluation with explicit LLM judge overrides
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
            raise_exceptions=False,
        )

        result_df = result.to_pandas()

        # Extract numeric mean scores while handling NaNs
        scores = {}
        for col in result_df.columns:
            if col in METRIC_MAP:
                mean_val = result_df[col].dropna().mean()
                scores[col] = float(mean_val) if not np.isnan(mean_val) else 0.0

        logger.info(f"RAGAS evaluation complete. Scores: {scores}")

        # Check CI gate (faithfulness threshold)
        faithfulness_score = scores.get("faithfulness", 0.0)
        passed = faithfulness_score >= self.cfg.groundedness_threshold

        if not passed:
            logger.warning(
                f"Evaluation Gate Failed: faithfulness {faithfulness_score:.3f} "
                f"below required threshold {self.cfg.groundedness_threshold:.3f}"
            )

        return {
            "scores": scores,
            "passed": passed,
            "detailed_results": result_df.to_dict(orient="records"),
        }