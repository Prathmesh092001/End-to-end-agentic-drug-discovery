# Partex Drug Asset Intelligence Copilot

`End-to-end-agentic-drug-discovery`

An end-to-end agentic GenAI pipeline for drug asset intelligence and
multi-modal drug discovery, built for the Partex Generative AI Engineer
role: MCP-standardized tool access, LangGraph multi-agent orchestration,
RAG over scientific data, RAGAS/LLM-as-judge evaluation, FastAPI serving,
and MLflow-tracked LLMOps with Docker + GitHub Actions CI/CD to AWS.

## Architecture

```
retrieval -> [profiling, risk] -> hypothesis -> competitive_intel -> critic -> report
```

7 agents: supervisor (LangGraph), retrieval, profiling, risk scoring,
hypothesis generation, competitive intelligence, and a critic/evaluation
agent that gates the final report on groundedness.

## Data sources (via MCP)

| Source | MCP server | Used by |
|---|---|---|
| PubChem | `mcp_servers/pubchem_mcp.py` | profiling agent |
| ChEMBL | `mcp_servers/chembl_mcp.py` | risk agent |
| UniProt | `mcp_servers/uniprot_mcp.py` | hypothesis agent |
| RCSB PDB | `mcp_servers/pdb_mcp.py` | profiling agent |
| Tavily (web) | `mcp_servers/tavily_mcp.py` | competitive intel agent |

## Setup (free-tier stack)

This project defaults to a fully free stack: **Groq** for the LLM,
**local HuggingFace embeddings**, and a **local cross-encoder reranker**
- only two free signups are needed.

```bash
conda create -n partex-agent python=3.11 -y
conda activate partex-agent
pip install -r requirements.txt
cp .env.example .env
# fill in GROQ_API_KEY (free at console.groq.com)
# fill in TAVILY_API_KEY (free tier at tavily.com)
```

To switch to a paid provider later (e.g. Anthropic Claude for higher
quality), just change `llm.provider` and `llm.model_name` in
`config/config.yaml` and uncomment the matching package in
`requirements.txt` - no code changes needed anywhere else.

You'll also need a running Qdrant instance for the vector store:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Run the full pipeline

```bash
python main.py
```

Runs: ingest internal KB -> chunk & index into Qdrant -> run the agent
graph on a sample query -> log the run to MLflow.

## Run the API

```bash
python app.py
# POST http://localhost:8000/v1/drug-intelligence
# {"query": "risk profile?", "compound_or_target": "imatinib"}
```

## MLflow tracking

```bash
mlflow ui
# or point MLFLOW_TRACKING_URI at DagsHub, as in config/config.yaml
```

## Tests

```bash
pytest tests/ -v
```

## Deployment

`Dockerfile` + `.github/workflows/cicd.yaml` build the image, push to AWS
ECR, and deploy to an EC2 self-hosted runner - see the reference repo's
AWS setup steps (IAM user, ECR repo, EC2 + Docker, GitHub Actions runner,
GitHub secrets) for the infra side; the workflow file here assumes that
infra already exists.

## Project layout

```
config/config.yaml          # static config: MCP endpoints, vector store, LLM
params.yaml                  # per-agent runtime params
src/partex_agent/
  entity/                    # typed config dataclasses
  config/                    # ConfigurationManager
  mcp_servers/               # one MCP server per data source + unified client
  components/retriever/      # chunking, vector store, reranker, retrieval agent
  components/agents/         # profiling, risk, hypothesis, competitive intel, critic
  components/orchestrator/   # LangGraph supervisor graph
  evaluation/                # RAGAS + LLM-as-judge
  pipeline/                  # stage_01_ingest, stage_02_index, stage_03_agent_run
app.py                       # FastAPI entrypoint
main.py                      # full pipeline orchestration
```
