# ==============================================================================
# WHAT: Realistic Multi-Plane GCP & Dynatrace Incident Dataset Generator
# WHY:  To build, test, and train our Stats [02], Pandas Correlator [03],
#       LightGBM [04], PyTorch DL [05], RAG [07], and MCP Agent [08] offline at
#       zero cloud cost, we need correlated cross-service incident scenarios
#       spanning BOTH Service-Log failures AND Data-Plane anomalies.
# HOW:  Generates 4 distinct production incidents (with shared `incident_id` tags
#       and realistic timestamps) plus healthy baseline telemetry across 7 JSON
#       files in `data/raw_logs/`.
# WHERE: `src/connectors/seed_incidents.py` (run once via CLI to populate data).
# EXAMPLE:
#   python src/connectors/seed_incidents.py
# ==============================================================================

import json                                                                 # Imports JSON library to write formatted log files
from pathlib import Path                                                    # Imports Path to ensure output directory exists


def generate_seed_logs(output_dir: str = "data/raw_logs") -> None:          # Main generator function creating all 7 JSON telemetry files
    out = Path(output_dir)                                                  # Wraps output directory string in a Path object
    out.mkdir(parents=True, exist_ok=True)                                  # Creates `data/raw_logs/` if it does not already exist

    # 1. Dynatrace Alerts (Alert Plane)
    dynatrace_alerts = [                                                    # List of Dynatrace Davis AI detected problem events
        {                                                                   # Incident 1: Service-Log Plane failure (Dataflow OOM -> Composer fail)
            "timestamp": "2026-11-27T02:15:00Z",                            # Alert fired 5 mins after Dataflow worker OOM began
            "resource_name": "df-clickstream-enrich-prod",                  # Impacted Dataflow streaming job
            "severity": "CRITICAL",                                         # High-priority Dynatrace problem severity
            "metric_name": "p99_processing_latency_ms",                     # Latency metric monitored by Dynatrace
            "metric_value": 18400.0,                                        # Spiked to 18.4 seconds
            "baseline_value": 410.0,                                        # Normal baseline was 410 ms
            "message": "Dynatrace Problem P-261101: Severe latency spike & worker restarts on df-clickstream-enrich-prod",
            "metadata": {"incident_id": "INC-1001", "source_tool": "dynatrace", "env": "prod"},
        },
        {                                                                   # Incident 2: Data-Plane failure (Silent BigQuery null spike)
            "timestamp": "2026-11-27T04:32:00Z",                            # Alert fired after morning dbt transformation completed
            "resource_name": "analytics.fact_daily_settlements",            # Impacted BigQuery gold table
            "severity": "CRITICAL",                                         # Critical business data corruption alert
            "metric_name": "null_rate_pct",                                 # Data-quality metric tracked via custom DQ monitor
            "metric_value": 41.8,                                           # 41.8% of settlement_amount rows became NULL
            "baseline_value": 0.05,                                         # Normal baseline null rate is 0.05%
            "message": "Dynatrace Problem P-261102: Data quality SLA breach on analytics.fact_daily_settlements (null_rate 41.8%)",
            "metadata": {"incident_id": "INC-1002", "source_tool": "dynatrace", "env": "prod"},
        },
    ]

    # 2. GCP Cloud Monitoring Alerts (Alert Plane)
    gcp_monitoring_alerts = [                                               # List of native Google Cloud Alerting policy triggers
        {                                                                   # Incident 3: Data-Plane + Infra hybrid (BQ Slot Quota + Row drop)
            "timestamp": "2026-11-27T06:10:00Z",                            # Native GCP alert timestamp
            "resource_name": "bq-proj-prod:finance_mart",                   # Impacted BigQuery dataset/project
            "severity": "ERROR",                                            # Alert severity
            "metric_name": "query_slot_wait_ms",                            # BQ slot contention metric
            "metric_value": 9200.0,                                         # Queries waiting 9.2 seconds for slots
            "baseline_value": 250.0,                                        # Normal slot wait is 250 ms
            "message": "GCP Alert Policy [BQ-SLA-04]: BigQuery concurrent slot quota exceeded; finance_mart refresh delayed",
            "metadata": {"incident_id": "INC-1003", "source_tool": "gcp_monitoring", "env": "prod"},
        },
        {                                                                   # Incident 4: Streaming Data-Plane backlog (Pub/Sub + Schema drift)
            "timestamp": "2026-11-27T08:05:00Z",                            # Native GCP alert timestamp
            "resource_name": "sub-orders-events-v2",                        # Impacted Pub/Sub subscription
            "severity": "CRITICAL",                                         # Critical streaming lag alert
            "metric_name": "oldest_unacked_message_age_sec",                # Pub/Sub backlog age metric
            "metric_value": 2450.0,                                         # Messages stuck unacked for > 40 minutes
            "baseline_value": 12.0,                                         # Normal backlog age is 12 seconds
            "message": "GCP Alert Policy [PS-LAG-01]: Pub/Sub subscription sub-orders-events-v2 backlog surging; DLQ spike detected",
            "metadata": {"incident_id": "INC-1004", "source_tool": "gcp_monitoring", "env": "prod"},
        },
    ]

    # 3. Cloud Composer / Airflow Logs (Service Log Plane)
    composer_logs = [                                                       # Airflow DAG & task execution logs
        {                                                                   # Healthy baseline run for comparison
            "timestamp": "2026-11-27T01:00:00Z",
            "resource_name": "dag_clickstream_hourly",
            "severity": "INFO",
            "metric_name": "task_duration_sec",
            "metric_value": 185.0,
            "baseline_value": 180.0,
            "message": "Task run_beam_enrichment succeeded in 185s. Rows processed: 1,420,110.",
            "metadata": {"incident_id": "NONE", "dag_id": "dag_clickstream_hourly", "task_id": "run_beam_enrichment"},
        },
        {                                                                   # Linked to INC-1001 (Composer sensor failed because Dataflow crashed)
            "timestamp": "2026-11-27T02:14:20Z",
            "resource_name": "dag_clickstream_hourly",
            "severity": "ERROR",
            "metric_name": "task_duration_sec",
            "metric_value": 1820.0,
            "baseline_value": 180.0,
            "message": "AirflowException: DataflowJobStatusSensor timed out waiting for df-clickstream-enrich-prod (state=JOB_STATE_FAILED)",
            "metadata": {"incident_id": "INC-1001", "dag_id": "dag_clickstream_hourly", "task_id": "wait_for_dataflow"},
        },
        {                                                                   # Linked to INC-1002 (dbt test failed inside Composer DAG)
            "timestamp": "2026-11-27T04:30:10Z",
            "resource_name": "dag_finance_dbt_daily",
            "severity": "ERROR",
            "metric_name": "task_duration_sec",
            "metric_value": 410.0,
            "baseline_value": 390.0,
            "message": "BashOperator failed: dbt test --select fact_daily_settlements exited with code 1 (not_null_settlement_amount FAILED)",
            "metadata": {"incident_id": "INC-1002", "dag_id": "dag_finance_dbt_daily", "task_id": "dbt_test_gold"},
        },
    ]

    # 4. Dataflow Logs (Service Log Plane)
    dataflow_logs = [                                                       # Apache Beam / Dataflow worker telemetry
        {                                                                   # Linked to INC-1001 (Root cause: OOM after large shuffle join)
            "timestamp": "2026-11-27T02:10:05Z",
            "resource_name": "df-clickstream-enrich-prod",
            "severity": "CRITICAL",
            "metric_name": "worker_heap_used_pct",
            "metric_value": 98.7,
            "baseline_value": 64.0,
            "message": "java.lang.OutOfMemoryError: GC overhead limit exceeded at org.apache.beam.sdk.transforms.GroupByKey (worker-03)",
            "metadata": {"incident_id": "INC-1001", "machine_type": "n2-standard-2", "workers": 4},
        },
        {                                                                   # Linked to INC-1004 (Dataflow crashing on unexpected JSON schema field)
            "timestamp": "2026-11-27T08:01:15Z",
            "resource_name": "df-orders-stream-ingest",
            "severity": "ERROR",
            "metric_name": "system_watermark_lag_sec",
            "metric_value": 2410.0,
            "baseline_value": 8.0,
            "message": "org.apache.beam.sdk.coders.CoderException: Cannot decode field 'currency_fx_rate': expected FLOAT64, got STRING 'N/A'",
            "metadata": {"incident_id": "INC-1004", "subscription": "sub-orders-events-v2"},
        },
    ]

    # 5. BigQuery & dbt Data Quality Signals (Data Plane)
    bigquery_dq_signals = [                                                 # Data-plane quality & job audit events
        {                                                                   # Linked to INC-1002 (Upstream raw table had 0 rows for FX lookup)
            "timestamp": "2026-11-27T04:25:00Z",
            "resource_name": "raw_erp.currency_exchange_rates",
            "severity": "CRITICAL",
            "metric_name": "row_count",
            "metric_value": 0.0,
            "baseline_value": 1450.0,
            "message": "Data Anomaly: Partition 2026-11-27 in raw_erp.currency_exchange_rates loaded 0 rows (expected ~1,450); LEFT JOIN produced NULLs",
            "metadata": {"incident_id": "INC-1002", "downstream_table": "analytics.fact_daily_settlements"},
        },
        {                                                                   # Linked to INC-1003 (Ad-hoc unpartitioned query exhausted BQ slots)
            "timestamp": "2026-11-27T06:06:40Z",
            "resource_name": "bq-proj-prod:finance_mart",
            "severity": "ERROR",
            "metric_name": "bytes_billed_tb",
            "metric_value": 48.6,
            "baseline_value": 1.2,
            "message": "Resources exceeded during query execution: BigQuery reservation prod-etl-slots saturated by full-table scan on raw_events",
            "metadata": {"incident_id": "INC-1003", "job_id": "bquxjob_77a91f_20261127"},
        },
    ]

    # 6. Pub/Sub Streaming Signals (Data Plane)
    pubsub_signals = [                                                      # Pub/Sub backlog & Dead-Letter Queue events
        {                                                                   # Linked to INC-1004
            "timestamp": "2026-11-27T08:03:00Z",
            "resource_name": "sub-orders-events-v2",
            "severity": "ERROR",
            "metric_name": "dlq_messages_per_min",
            "metric_value": 1240.0,
            "baseline_value": 0.0,
            "message": "Dead-letter topic orders-events-dlq receiving 1,240 msgs/min due to schema mismatch in producer v2.14",
            "metadata": {"incident_id": "INC-1004", "dlq_topic": "orders-events-dlq"},
        }
    ]

    # 7. Harness CI/CD & Terraform State Events (CI/CD Plane)
    harness_deploys = [                                                     # Deployment & IaC change audit trail
        {                                                                   # Linked to INC-1001 (Harness deployed larger window size 12 mins prior)
            "timestamp": "2026-11-27T01:58:00Z",
            "resource_name": "harness-deploy-clickstream-beam",
            "severity": "INFO",
            "metric_name": "deploy_duration_sec",
            "metric_value": 95.0,
            "baseline_value": 90.0,
            "message": "Harness CD Pipeline succeeded: Deployed image gcr.io/prod/clickstream-beam:v4.19.0 (changed session window 5m -> 60m)",
            "metadata": {"incident_id": "INC-1001", "git_commit": "a94f21c", "author": "data-eng-team"},
        },
        {                                                                   # Linked to INC-1004 (Producer service deployed breaking schema)
            "timestamp": "2026-11-27T07:55:00Z",
            "resource_name": "harness-deploy-checkout-api",
            "severity": "INFO",
            "metric_name": "deploy_duration_sec",
            "metric_value": 82.0,
            "baseline_value": 80.0,
            "message": "Harness CD Pipeline succeeded: Deployed checkout-api:v2.14.0 (serialized currency_fx_rate as string 'N/A' when missing)",
            "metadata": {"incident_id": "INC-1004", "git_commit": "f102b8e", "author": "checkout-team"},
        },
    ]

    datasets = {                                                            # Maps output JSON filenames to their generated event lists
        "dynatrace_alerts.json": dynatrace_alerts,
        "gcp_monitoring_alerts.json": gcp_monitoring_alerts,
        "composer_logs.json": composer_logs,
        "dataflow_logs.json": dataflow_logs,
        "bigquery_dq_signals.json": bigquery_dq_signals,
        "pubsub_signals.json": pubsub_signals,
        "harness_deploys.json": harness_deploys,
    }

    for filename, records in datasets.items():                              # Loops over all 7 files and writes clean formatted JSON
        target = out / filename                                             # Builds full output path inside `data/raw_logs/`
        with open(target, "w", encoding="utf-8") as f:                      # Opens target JSON file in UTF-8 write mode
            json.dump(records, f, indent=2)                                 # Serializes records with 2-space indentation
        print(f"Generated {target} ({len(records)} signals)")               # Prints confirmation count for each generated dataset


if __name__ == "__main__":                                                  # Standard Python entry-point guard when run directly from terminal
    generate_seed_logs()                                                    # Executes dataset generation into `data/raw_logs/`