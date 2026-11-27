# omni-rca-gcp
Plug-and-play GCP Incident Triage &amp; RCA Copilot (Stats, ML, PyTorch DL, ChromaDB RAG, and MCP Agent)
# OmniRCA-GCP: Plug-and-Play GCP Autonomous Incident Triage & RCA Copilot (v1.0)

> **A local-first, zero-cost AI Engineering & Data Science project** that ingests both **Service/Log-Plane** and **Data-Plane** signals across the Google Cloud Platform ecosystem (Cloud Composer, Dataflow, BigQuery, Pub/Sub, Cloud Monitoring/Alerting, dbt, Terraform, Harness), mathematically validates anomalies, predicts downstream SLA cascade risk, classifies stack traces with PyTorch, retrieves historical fixes from a self-learning Vector DB, orchestrates Root Cause Analysis (RCA) via an MCP Server and Streamlit UI, and passively curates an instruction-tuning dataset for a future custom GCP RCA model.

---

## 1. Business Use Case & Problem Statement

### The Operational Bottleneck
In modern GCP data platforms orchestrated by **Cloud Composer (Airflow)**, **Dataflow**, **BigQuery**, **Pub/Sub**, **dbt**, **Terraform**, and **Harness**, incidents occur across **two distinct planes**:
1. **Service & Infrastructure Plane:** Worker OOM crashes, Airflow DAG timeouts, Pub/Sub unacked backlogs, Terraform state locks, or broken Harness deployments.
2. **Data Plane:** Silent null-rate spikes in BigQuery tables, sudden row-count drops after a dbt run, upstream schema drift, or stale partition watermarks.

Today, whenever an incident alert triggers (via Cloud Monitoring, Dynatrace, PagerDuty, or manual engineer triage), on-call engineers must manually jump across Cloud Logging, BigQuery `INFORMATION_SCHEMA` / job histories, Composer task logs, and Harness release histories—and then search past tickets to recall how similar issues were resolved.

### What OmniRCA-GCP Delivers
* **Version 1.0 (Current Build — Full-Stack RCA, Self-Learning Vector Memory & Plug-and-Play UI):**
  * **Plug-and-Play Selective Ingestion:** Engineers toggle **only** the connectors relevant to an incident (e.g., `[x] Cloud Composer`, `[x] Dataflow`, `[x] BigQuery Data Quality`, `[ ] Pub/Sub`, `[ ] Harness/Terraform`) via the UI—eliminating log noise and token bloat, while allowing new connectors to be added in a single file.
  * **Multi-Layer Triage Pipeline:** Combines statistical anomaly testing, cross-service timestamp correlation, LightGBM + SHAP cascade risk prediction, PyTorch stack-trace classification, and ChromaDB vector retrieval.
  * **Custom-GPT Conversational Interface & Model-Training Flywheel:** Lets engineers ask follow-up questions grounded in active signals, saves verified RCAs back into ChromaDB Vector Memory, and exports clean JSONL instruction pairs (`data/fine_tune_dataset/rca_pairs.jsonl`) to train a dedicated fine-tuned GCP RCA model in the future.
* **Version 2.0 (Planned Enhancement — Human-in-the-Loop Auto-Heal & Custom Model):**
  * Extends the MCP Server with write-action tools (`trigger_composer_dag_retry`, `scale_dataflow_workers`, `open_harness_rollback_pr`) gated by engineer approval, powered by our fine-tuned domain model.

---

## 2. System Architecture

```text
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT ENGINEER CONSOLE (ui/app.py)                        │
│ [x] Alerts  [x] Composer  [x] Dataflow  [x] BigQuery DQ  [ ] Pub/Sub  [ ] Harness/TF  │
│  • Select Incident ID    • Ask Custom-GPT Follow-ups    • "Approve & Save RCA"        │
└───────────────────────────────────────────┬───────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                   MCP SERVER & TRIAGE AGENT (src/agent_mcp/)                          │
│  Exposes Plug-and-Play Tools via Official MCP Python SDK:                             │
│   • tool: fetch_service_signals(service_name, resource_filter)                        │
│   • tool: run_statistical_and_ml_diagnostics(incident_id)                             │
│   • tool: search_vector_rca_memory(error_or_dq_signature)                             │
│   • tool: persist_verified_rca(rca_payload)  ──► ChromaDB + Fine-Tune JSONL Export    │
└───────────┬───────────────────────────────┬───────────────────────────────┬───────────┘
            │                               │                               │
            ▼                               ▼                               ▼
┌───────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────┐
│ PLUG-AND-PLAY         │   │ INTELLIGENCE PIPELINE         │   │ VECTOR KNOWLEDGE      │
│ CONNECTORS            │   │ (src/intelligence/)           │   │ (src/knowledge/)      │
│ (src/connectors/)     │   │ • Math/Stats: Welch t-test &  │   │ • ChromaDB Local      │
│ • Incident Alerts     │   │   KL-Divergence [02]          │   │   Persistent Store    │
│ • Cloud Composer      │   │ • Analytics: Cross-plane time │   │ • Local Sentence      │
│ • Cloud Dataflow      │   │   window joiner [03]          │   │   Embeddings (Free)   │
│ • BigQuery & dbt DQ   │   │ • ML: LightGBM + SHAP [04]    │   └───────────┬───────────┘
│ • Pub/Sub Streaming   │   │ • DL: PyTorch Stack-Trace     │               │
│ • Harness & Terraform │   │   1D-CNN Classifier [05]      │               │
└───────────────────────┘   └───────────────────────────────┘               │
                                            │                               │
                                            ▼                               ▼
                             ┌─────────────────────────────────────────────────────────┐
                             │ STRUCTURED GENAI SYNTHESIZER (src/llm_engine/) [06]     │
                             │ • Free/Local LLM (Groq / DeepSeek / Qwen / Ollama)      │
                             │ • Pydantic Enforced Output:                             │
                             │   1. Root Cause  2. Timeline  3. Exact Fix Commands     │
                             └─────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure & Learning Syllabus Map (`02`–`09`)

This repository uses an **industry-standard modular layout** rather than tutorial-numbered folders. Every core module maps directly to your learning syllabus (`02` through `09`):

| Syllabus Topic | Production File(s) | Core Concepts & Algorithms Implemented |
| :--- | :--- | :--- |
| **`02_Mathematics_and_Statistics`** | `src/intelligence/stats_drift.py` | **Modified Z-Score (Median/MAD)** for outlier-resistant spike detection, **Welch’s Two-Sample t-test** for unequal-variance latency/watermark shifts, and **KL-Divergence** for data & latency distribution drift. |
| **`03_Data_Analysis`** | `src/intelligence/log_correlator.py` | Rolling SLA windows (`p95`/`p99`), null-rate/row-count delta profiling, and `pandas.merge_asof` temporal joins linking alerts to upstream Harness deploys, Dataflow crashes, and BigQuery anomalies. |
| **`04_Machine_Learning`** | `src/intelligence/cascade_ml.py` | **LightGBM Classifier** trained on cross-plane telemetry features to predict downstream BigQuery SLA breach probability, paired with **SHAP** feature attribution. |
| **`05_Deep_Learning`** | `src/intelligence/trace_dl.py` | Custom **PyTorch Token-Embedding + 1D-CNN** neural network trained locally to classify raw Java/Python/SQL error traces (`DATAFLOW_OOM`, `BQ_SCHEMA_OR_QUOTA`, `COMPOSER_TIMEOUT`, `PUBSUB_BACKLOG`, `TERRAFORM_LOCK`). |
| **`06_Generative_AI`** | `src/llm_engine/rca_generator.py` | System/user prompt engineering, multi-plane evidence grounding, and **Pydantic JSON Schema Structured Outputs** using free/local LLMs (Groq, DeepSeek, Qwen, or Ollama). |
| **`07_RAG_and_Vector_Databases`** | `src/knowledge/rca_vector_store.py` | Document chunking, local `sentence-transformers` embeddings (`all-MiniLM-L6-v2`), **ChromaDB** metadata filtering, semantic retrieval, self-learning RCA persistence, and JSONL training export. |
| **`08_Agentic_AI_and_MCP`** | `src/agent_mcp/mcp_server.py`<br>`src/agent_mcp/rca_agent.py` | Official **Model Context Protocol (MCP)** server exposing active connectors and diagnostic engines as tools, driven by an autonomous tool-calling triage agent. |
| **`09_End_to_End_Projects`** | `ui/app.py`<br>`src/connectors/base.py`<br>`configs/connectors.yaml` | Plugin registry pattern (`@ConnectorRegistry`), YAML-driven configuration, Pytest validation, and an interactive **Streamlit** engineer console. |

### Directory Tree

```text
omni-rca-gcp/
├── configs/
│   ├── connectors.yaml              # Plug-and-play toggles (Data Plane + Service Plane)
│   └── settings.yaml                # Model hyperparameters, ChromaDB path, LLM provider
├── data/
│   ├── raw_logs/                    # Realistic multi-service GCP JSON signals (logs + DQ)
│   ├── seed_runbooks/               # Initial engineering runbooks (Composer, Dataflow, BQ, TF)
│   ├── chroma_db/                   # Local persistent Vector DB storage
│   └── fine_tune_dataset/           # Auto-curated JSONL pairs for future custom model training
├── artifacts/                       # Saved trained LightGBM (.pkl) & PyTorch (.pt) models
├── src/
│   ├── connectors/                  # Plug-and-play ingestion layer
│   │   ├── base.py                  # Abstract BaseConnector + IncidentSignal schema + Registry
│   │   ├── alerts_conn.py           # Incident alert trigger connector (Monitoring / Dynatrace)
│   │   ├── gcp_composer.py          # Cloud Composer / Airflow log connector
│   │   ├── gcp_dataflow.py          # Dataflow worker & watermark connector
│   │   ├── gcp_bigquery_dq.py       # BigQuery & dbt Data Quality + job audit connector
│   │   └── gcp_pubsub_harness.py    # Pub/Sub backlog & Harness/Terraform CI/CD connectors
│   ├── intelligence/                # Core Math, Analytics, ML & Deep Learning engines
│   │   ├── stats_drift.py           # [Topic 02: Math & Stats]
│   │   ├── log_correlator.py        # [Topic 03: Data Analysis]
│   │   ├── cascade_ml.py            # [Topic 04: Machine Learning]
│   │   └── trace_dl.py              # [Topic 05: Deep Learning]
│   ├── knowledge/                   # RAG & Vector Memory
│   │   └── rca_vector_store.py      # [Topic 07: RAG & Vector Databases]
│   ├── llm_engine/                  # Structured Generative AI
│   │   └── rca_generator.py         # [Topic 06: Generative AI]
│   └── agent_mcp/                   # MCP Server & Autonomous Agent
│       ├── mcp_server.py            # [Topic 08: Agentic AI & MCP]
│       └── rca_agent.py             # Tool-calling investigation orchestrator
├── ui/
│   └── app.py                       # [Topic 09: End-to-End Projects] Streamlit Web UI
├── tests/                           # Pytest unit tests for math, connectors, and schemas
├── README.md
└── requirements.txt
```

---

## 4. 3-Month Agile Implementation Plan (6 Sprints / 18 Tickets)

### Sprint 1 (Weeks 1–2): Foundation & Plug-and-Play Connectors
- [x] **TICKET-101:** Initialize repository layout, `requirements.txt`, and `README.md` architecture map.
- [x] **TICKET-102:** Build `configs/connectors.yaml` and `src/connectors/base.py` (`IncidentSignal` Pydantic contract, `BaseConnector`, and `@ConnectorRegistry`).
- [ ] **TICKET-103:** Build multi-plane GCP incident dataset generator (covering both Service Logs and Data Quality failures) & concrete plug-and-play connectors.

### Sprint 2 (Weeks 3–4): Statistical Anomaly Math & Cross-Plane Analysis
- [ ] **TICKET-201 `[Topic 02]`:** Implement `src/intelligence/stats_drift.py` (Modified Z-Score, Welch’s t-test, KL-Divergence).
- [ ] **TICKET-202 `[Topic 03]`:** Implement `src/intelligence/log_correlator.py` (rolling SLA/DQ metrics & `merge_asof` cross-plane event correlation).
- [ ] **TICKET-203:** Add `tests/test_stats_and_correlator.py` unit tests.

### Sprint 3 (Weeks 5–6): Predictive ML & Deep Learning Trace Classifier
- [ ] **TICKET-301 `[Topic 04]`:** Build & train `src/intelligence/cascade_ml.py` (LightGBM SLA cascade classifier + SHAP explainability).
- [ ] **TICKET-302 `[Topic 05]`:** Build & train `src/intelligence/trace_dl.py` (PyTorch Token-Embedding + 1D-CNN root-cause classifier).
- [ ] **TICKET-303:** Validate model metrics and export local inference artifacts to `artifacts/`.

### Sprint 4 (Weeks 7–8): Self-Learning Vector Database & Hybrid RAG Memory
- [ ] **TICKET-401 `[Topic 07]`:** Author 8 seed GCP/DevOps runbooks in `data/seed_runbooks/`.
- [ ] **TICKET-402 `[Topic 07]`:** Build `src/knowledge/rca_vector_store.py` (ChromaDB + local `sentence-transformers` + metadata filtering).
- [ ] **TICKET-403:** Implement `save_solved_rca()` write-back memory loop + JSONL fine-tuning dataset exporter.

### Sprint 5 (Weeks 9–10): Structured GenAI Synthesizer & MCP Server
- [ ] **TICKET-501 `[Topic 06]`:** Build `src/llm_engine/rca_generator.py` (Pydantic `RCAReport` schema + free/local LLM API client).
- [ ] **TICKET-502 `[Topic 08]`:** Build `src/agent_mcp/mcp_server.py` exposing active connectors and diagnostics via official MCP SDK.
- [ ] **TICKET-503 `[Topic 08]`:** Build `src/agent_mcp/rca_agent.py` autonomous investigation loop.

### Sprint 6 (Weeks 11–12): Plug-and-Play Engineer Web UI & End-to-End Polish
- [ ] **TICKET-601 `[Topic 09]`:** Build `ui/app.py` (Streamlit sidebar connector toggles, incident selector, live diagnostic cards, Custom-GPT Q&A chat).
- [ ] **TICKET-602 `[Topic 09]`:** Wire one-click **"Approve & Save RCA to Vector Memory"** workflow.
- [ ] **TICKET-603:** End-to-end validation and v2.0 Auto-Heal + Custom Model Fine-Tuning guide.

---

## 5. Local Quickstart (Zero-Cost Setup)

```powershell
# 1. Activate your Anaconda Python environment and install dependencies
pip install -r requirements.txt

# 2. Run unit tests (available from Sprint 2 onward)
pytest tests/

# 3. Launch the Streamlit Engineer Console (Sprint 6)
streamlit run ui/app.py
```

### How to Plug In a New GCP Connector Later
1. Create a new file in `src/connectors/` (e.g., `gcp_cloud_run.py`).
2. Inherit from `BaseConnector`, decorate with `@ConnectorRegistry.register("cloud_run")`, and implement `fetch_signals()`.
3. Add a `cloud_run:` entry in `configs/connectors.yaml`—it automatically appears as a toggleable checkbox in the UI and as a callable tool in the MCP Server.