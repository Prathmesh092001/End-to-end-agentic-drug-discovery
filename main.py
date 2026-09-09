from partex_agent.pipeline.stage_01_ingest import run_ingestion_stage
from partex_agent.pipeline.stage_02_index import run_indexing_stage
from partex_agent.pipeline.stage_03_agent_run import run_agent_stage
from partex_agent.logging import logger

STAGE_NAME_1 = "Data ingestion stage"
STAGE_NAME_2 = "Knowledge base indexing stage"
STAGE_NAME_3 = "Agent orchestration stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME_1} started <<<<<<")
        run_ingestion_stage()
        logger.info(f">>>>>> {STAGE_NAME_1} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> {STAGE_NAME_2} started <<<<<<")
        run_indexing_stage()
        logger.info(f">>>>>> {STAGE_NAME_2} completed <<<<<<\n\nx==========x")

        logger.info(f">>>>>> {STAGE_NAME_3} started <<<<<<")
        run_agent_stage(
            query="Summarize the risk and opportunity profile of this compound.",
            compound_or_target="imatinib",
        )
        logger.info(f">>>>>> {STAGE_NAME_3} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e
