import os
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from partex_agent.config.configuration import ConfigurationManager
from partex_agent.pipeline.stage_03_agent_run import run_agent_stage
from partex_agent.logging import logger

app = FastAPI(
    title="Partex Drug Asset Intelligence Copilot",
    description="Agentic GenAI API for compound profiling, risk scoring, hypothesis generation, and competitive intelligence.",
    version="0.1.0",
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# Mount static files so http://localhost:8000 directly serves ui/public/index.html
app.mount("/", StaticFiles(directory="ui/public", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    try:
        serving_cfg = ConfigurationManager().get_serving_config()
        cfg_host = getattr(serving_cfg, "host", "0.0.0.0")
        cfg_port = int(getattr(serving_cfg, "port", 8000))
    except Exception as e:
        logger.warning(f"Could not load serving config, defaulting to 0.0.0.0:8000. Error: {e}")
        cfg_host = "0.0.0.0"
        cfg_port = 8000

    host = os.getenv("HOST", cfg_host)
    port = int(os.getenv("PORT", cfg_port))

    # reload=False prevents Uvicorn from restarting and wiping logs whenever the
    # 7-agent pipeline writes files to logs/, artifacts/, mlruns/, or qdrant_storage/.
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        reload_excludes=[
            "logs/*",
            "artifacts/*",
            "mlruns/*",
            "qdrant_storage/*",
            "*.log",
        ],
    )