Think of src/connectors/base.py as the universal adapter and switchboard for OmniRCA-GCP.

In a real GCP environment, Dynatrace sends nested JSON problem payloads, Airflow writes Python tracebacks, BigQuery outputs row-count and slot metrics, and Harness logs deployment events. Without a common standard, every downstream AI, ML, and statistical module would need custom if/else parsing logic for every single tool.

Here is how the three classes in base.py solve that cleanly:

1. IncidentSignal(BaseModel) — The Universal Data Contract
What it is: A strict Pydantic schema that defines what every piece of telemetry looks like once it enters OmniRCA-GCP.

Why we need it: Whether an incident starts from a service log (e.g., Dataflow OutOfMemoryError) or a data-plane anomaly (e.g., BigQuery null_rate_pct jumping from 0.05 to 41.8), both are converted into the exact same 10 fields (timestamp, service, plane, resource_name, severity, metric_name, metric_value, baseline_value, message, metadata).

Runtime safety: If a raw log has a string "41.8" instead of a float, Pydantic automatically validates and coerces it—or raises a clear validation error immediately rather than crashing your LightGBM or PyTorch model later.

2. BaseConnector(ABC) — The Plugin Blueprint (Contract)
What it is: An Abstract Base Class (ABC) that every connector plugin must inherit from.

Why we need it: By marking fetch_signals() with @abstractmethod, Python enforces a rule: no developer can create a new connector (for BigQuery, Composer, or PagerDuty tomorrow) without implementing .fetch_signals() and returning a List[IncidentSignal].

How it works: When a plugin initializes (__init__), it automatically receives its YAML key (service_key) and its dictionary block (config) from connectors.yaml.

3. ConnectorRegistry — The Plug-and-Play Switchboard
What it is: An implementation of the Registry Decorator Design Pattern using class-level state (_REGISTRY = {}).

How the 3 methods work together:

@ConnectorRegistry.register("dataflow"): When Python imports a plugin file, this decorator automatically registers that class inside _REGISTRY["dataflow"] = DataflowConnector. You never have to manually maintain a master list of classes.

list_available_configs(): Reads configs/connectors.yaml and returns all configured services so the Streamlit UI can dynamically draw checkboxes for every available tool.

load_connectors(selected_services=...): Cross-references the YAML config with what the engineer toggled ON in the UI (or falls back to enabled: true in YAML) and instantiates only the active connectors. Disabled tools consume zero memory and zero API calls.

How They Connect End-to-End (Mental Model)
Plaintext
configs/connectors.yaml  ──(read by)──►  ConnectorRegistry.load_connectors()
                                                │
                                   instantiates active plugins
                                                ▼
                                     DataflowConnector(BaseConnector)
                                                │
                                     calls .fetch_signals()
                                                ▼
                                     List[IncidentSignal]  ──►  Stats / ML / DL / RAG / MCP


Understanding with an Example:

Let’s trace a single real-world incident—INC-1002: A silent BigQuery data-quality failure triggered at 4:32 AM—from raw messy payloads all the way through ConnectorRegistry, BaseConnector, and IncidentSignal.1. The Raw Problem (Two Completely Different Payloads)At 4:32 AM, two systems emit events in totally different shapes:Dynatrace raw alert JSON:JSON{
  "problemId": "P-261102",
  "startTime": "2026-11-27T04:32:00Z",
  "impactedEntity": "analytics.fact_daily_settlements",
  "severityLevel": "CUSTOM_ALERT",
  "metricValue": "41.8"
}
BigQuery Data Quality check JSON:JSON{
  "ts": "2026-11-27T04:25:00Z",
  "table_id": "raw_erp.currency_exchange_rates",
  "rows_loaded": 0,
  "expected_rows": 1450,
  "alert_text": "Partition 2026-11-27 loaded 0 rows; LEFT JOIN produced NULLs"
}
Notice the mismatch: Dynatrace calls the timestamp "startTime" and passes "41.8" as a string, while BigQuery calls it "ts" and passes "rows_loaded": 0.2. Step-by-Step Walkthrough Through base.pyStep A: @ConnectorRegistry.register wires up the switchboard at import timeWhen Python loads your connector plugins, the decorator runs once per class:Python@ConnectorRegistry.register("dynatrace")
class DynatraceConnector(BaseConnector): ...

@ConnectorRegistry.register("bigquery_data_quality")
class BigQueryDQConnector(BaseConnector): ...
Behind the scenes, ConnectorRegistry._REGISTRY now holds:Python{
    "dynatrace": DynatraceConnector,
    "bigquery_data_quality": BigQueryDQConnector,
    # ...other registered plugins
}
Step B: ConnectorRegistry.load_connectors() activates ONLY what you toggled ONSuppose in the Streamlit UI you check only dynatrace and bigquery_data_quality (leaving pubsub_streaming and harness_terraform unchecked):Pythonactive = ConnectorRegistry.load_connectors(
    selected_services=["dynatrace", "bigquery_data_quality"]
)
load_connectors() calls list_available_configs() to read configs/connectors.yaml.It loops through all 7 YAML entries, sees that only those 2 keys are in selected_services, and instantiates only those 2 classes—passing each its YAML dictionary (config).active now holds:Python{
    "dynatrace": <DynatraceConnector instance>,
    "bigquery_data_quality": <BigQueryDQConnector instance>
}
(The other 5 connectors aren't even instantiated in memory.)Step C: BaseConnector guarantees .fetch_signals() exists on every pluginBecause both classes inherit from BaseConnector(ABC), Python guarantees both have .fetch_signals(). If someone writes a new PagerDutyConnector(BaseConnector) tomorrow and forgets def fetch_signals(...), Python immediately raises TypeError: Can't instantiate abstract class PagerDutyConnector without an implementation for abstract method 'fetch_signals' before the app even runs.Step D: IncidentSignal normalizes both payloads into identical objectsWhen the agent loops over active.values() and calls connector.fetch_signals(), both plugins output identical IncidentSignal objects:Output 1 (from DynatraceConnector):PythonIncidentSignal(
    timestamp="2026-11-27T04:32:00Z",
    service="dynatrace",
    plane="alert",
    resource_name="analytics.fact_daily_settlements",
    severity="CRITICAL",
    metric_name="null_rate_pct",
    metric_value=41.8,          # <-- Pydantic automatically coerced string "41.8" into float 41.8!
    baseline_value=0.05,
    message="Data quality SLA breach on analytics.fact_daily_settlements (null_rate 41.8%)",
    metadata={"incident_id": "INC-1002"}
)
Output 2 (from BigQueryDQConnector):PythonIncidentSignal(
    timestamp="2026-11-27T04:25:00Z",
    service="bigquery_data_quality",
    plane="data_quality",
    resource_name="raw_erp.currency_exchange_rates",
    severity="CRITICAL",
    metric_name="row_count",
    metric_value=0.0,
    baseline_value=1450.0,
    message="Partition 2026-11-27 loaded 0 rows; LEFT JOIN produced NULLs",
    metadata={"incident_id": "INC-1002"}
)
3. Why This Pays Off Immediately DownstreamBecause both signals now share the exact same IncidentSignal structure, your downstream Pandas correlator ([Topic 03]) can sort both lists by .timestamp in one line and immediately see:04:25:00 UTC: raw_erp.currency_exchange_rates loaded 0.0 rows (vs 1450.0 baseline) $\rightarrow$ 7 minutes later at 04:32:00 UTC: Dynatrace fired a 41.8% null-rate alert on analytics.fact_daily_settlements.