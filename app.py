import os
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from partex_agent.config.configuration import ConfigurationManager
from partex_agent.logging import logger
from partex_agent.pipeline.stage_03_agent_run import run_agent_stage

# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui" / "public"


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Partex Drug Asset Intelligence Copilot",
    description=(
        "Agentic GenAI API for compound profiling, risk scoring, "
        "hypothesis generation, and competitive intelligence."
    ),
    version="0.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request / Response Models
# ============================================================

class IntelligenceRequest(BaseModel):
    query: str
    compound_or_target: str


class IntelligenceResponse(BaseModel):
    final_report: str
    approved: bool
    groundedness_score: float


# ============================================================
# Helper Functions
# ============================================================

def _safe_extract_groundedness(verdict: Union[Dict[str, Any], List[Any], None]) -> float:
    """Safely extracts groundedness score preventing IndexError or TypeError."""
    if not verdict:
        return 0.0

    if isinstance(verdict, list):
        verdict = verdict[0] if len(verdict) > 0 else {}

    if isinstance(verdict, dict):
        score = verdict.get("groundedness_score", 0.0)
        if isinstance(score, list):
            score = score[0] if len(score) > 0 else 0.0
        try:
            return float(score)
        except (ValueError, TypeError):
            return 0.0

    return 0.0


def _safe_extract_approval(verdict: Union[Dict[str, Any], List[Any], None]) -> bool:
    """Safely extracts approval boolean preventing indexing errors."""
    if not verdict:
        return False

    if isinstance(verdict, list):
        verdict = verdict[0] if len(verdict) > 0 else {}

    if isinstance(verdict, dict):
        return bool(verdict.get("approved", False))

    return False


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# Drug Intelligence API
# ============================================================

@app.post(
    "/v1/drug-intelligence",
    response_model=IntelligenceResponse,
)
def get_drug_intelligence(request: IntelligenceRequest):
    try:
        result = run_agent_stage(
            query=request.query,
            compound_or_target=request.compound_or_target,
        )
    except IndexError as ie:
        logger.exception("Agent pipeline failed due to empty evidence list indexing")
        raise HTTPException(
            status_code=500,
            detail=(
                "Pipeline encountered an index boundary error while parsing empty evidence/tool results. "
                f"Details: {str(ie)}"
            ),
        )
    except Exception as e:
        logger.exception("agent pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    # Ensure result is a dict
    if not isinstance(result, dict):
        result = {}

    verdict = result.get("critic_verdict", {}) or {}

    report = result.get("final_report") or result.get("report") or "*No report produced by the pipeline.*"
    if isinstance(report, list):
        report = "\n\n".join(str(item) for item in report) if report else "*Empty report generated.*"

    approved = _safe_extract_approval(verdict)
    groundedness_score = _safe_extract_groundedness(verdict)

    return IntelligenceResponse(
        final_report=str(report),
        approved=approved,
        groundedness_score=groundedness_score,
    )


# ============================================================
# Redesigned UI
# ============================================================

if not UI_DIR.exists():
    logger.warning(f"UI directory not found: {UI_DIR}")
else:
    logger.info(f"Serving redesigned UI from: {UI_DIR}")

    app.mount(
        "/",
        StaticFiles(
            directory=str(UI_DIR),
            html=True,
        ),
        name="static",
    )


# ============================================================
# Application Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn

    try:
        serving_cfg = ConfigurationManager().get_serving_config()
        cfg_host = getattr(serving_cfg, "host", "0.0.0.0")
        cfg_port = int(getattr(serving_cfg, "port", 8000))
    except Exception as e:
        logger.warning(
            "Could not load serving config, "
            f"defaulting to 0.0.0.0:8000. Error: {e}"
        )
        cfg_host = "0.0.0.0"
        cfg_port = 8000

    host = os.getenv("HOST", cfg_host)
    port = int(os.getenv("PORT", cfg_port))

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