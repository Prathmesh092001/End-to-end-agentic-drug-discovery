from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from partex_agent.config.configuration import ConfigurationManager
from partex_agent.pipeline.stage_03_agent_run import run_agent_stage
from partex_agent.logging import logger

app = FastAPI(
    title="Partex Drug Asset Intelligence Copilot",
    description="Agentic GenAI API for compound profiling, risk scoring, hypothesis generation, and competitive intelligence.",
    version="0.1.0",
)


class IntelligenceRequest(BaseModel):
    query: str
    compound_or_target: str


class IntelligenceResponse(BaseModel):
    final_report: str
    approved: bool
    groundedness_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/drug-intelligence", response_model=IntelligenceResponse)
def get_drug_intelligence(request: IntelligenceRequest):
    try:
        result = run_agent_stage(query=request.query, compound_or_target=request.compound_or_target)
    except Exception as e:
        logger.exception("agent pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))

    verdict = result.get("critic_verdict", {}) or {}
    return IntelligenceResponse(
        final_report=result.get("final_report", ""),
        approved=verdict.get("approved", False),
        groundedness_score=verdict.get("groundedness_score", 0.0),
    )


if __name__ == "__main__":
    import uvicorn

    serving_cfg = ConfigurationManager().get_serving_config()
    uvicorn.run("app:app", host=serving_cfg.host, port=serving_cfg.port, reload=True)
