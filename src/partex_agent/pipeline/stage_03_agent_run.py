from pathlib import Path
import mlflow

from partex_agent.config.configuration import ConfigurationManager
from partex_agent.components.orchestrator.graph import run_pipeline
from partex_agent.utils.common import timed
from partex_agent.logging import logger


@timed
def run_agent_stage(query: str, compound_or_target: str) -> dict:
    config_manager = ConfigurationManager()
    mlflow_cfg = config_manager.get_mlflow_config()

    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)

    with mlflow.start_run(run_name=f"query::{compound_or_target}"):
        mlflow.log_param("query", query)
        mlflow.log_param("compound_or_target", compound_or_target)

        final_state = run_pipeline(query=query, compound_or_target=compound_or_target)

        verdict = final_state.get("critic_verdict", {})
        mlflow.log_metric("groundedness_score", verdict.get("groundedness_score", 0.0))
        mlflow.log_metric("approved", int(verdict.get("approved", False)))
        
        final_report_content = final_state.get("final_report", "")
        
        # Log to MLflow artifacts
        mlflow.log_text(final_report_content, "final_report.md")

        # Save directly to local artifacts directory
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        report_file = artifacts_dir / "final_report.md"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(final_report_content)

        logger.info(f"Saved final report locally to {report_file}")

    logger.info("agent run stage complete, logged to MLflow")
    return final_state


if __name__ == "__main__":
    result = run_agent_stage(
        query="What is the risk and opportunity profile of this compound?",
        compound_or_target="metformin",
    )
    print(result.get("final_report"))