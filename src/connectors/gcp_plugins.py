# ==============================================================================
# WHAT: Concrete Plug-and-Play Connectors for Alerts, GCP Services, Data Quality & CI/CD
# WHY:  Each monitoring/GCP tool (Dynatrace, Cloud Monitoring, Composer, Dataflow,
#       BigQuery DQ, Pub/Sub, Harness/Terraform) stores raw events in its own JSON
#       structure. Wrapping each in a dedicated `@ConnectorRegistry.register` plugin
#       keeps ingestion modular—and lets you add new connectors (e.g., PagerDuty,
#       Cloud Run) in ~15 lines without touching any downstream ML/RAG/MCP code.
# HOW:  Each plugin subclasses `BaseConnector`, reads its configured JSON path (or
#       future live GCP/Dynatrace REST API), filters by `resource_filter` if given,
#       and returns validated `IncidentSignal` objects.
# WHERE: Imported by `src/connectors/__init__.py` so all plugins auto-register at
#        startup when `ConnectorRegistry.load_connectors()` is called.
# EXAMPLE:
#   from src.connectors import ConnectorRegistry
#   active = ConnectorRegistry.load_connectors(selected_services=["dynatrace", "dataflow"])
#   signals = active["dataflow"].fetch_signals()
# ==============================================================================

import json                                                                 # Imports standard JSON parser to read local raw telemetry files
from pathlib import Path                                                    # Imports Path for clean, cross-platform Windows/Linux file paths
from typing import List, Optional                                           # Imports type hints for lists and optional string arguments
from src.connectors.base import BaseConnector, ConnectorRegistry, IncidentSignal # Imports our base contract, registry, and canonical model


class LocalJsonBasePlugin(BaseConnector):                                   # Shared helper parent class for connectors reading local JSON logs
    def _load_raw_records(self) -> List[dict]:                              # Internal helper method to safely read JSON arrays from disk
        file_path = Path(self.config.get("path", ""))                       # Retrieves the file path configured for this service in YAML
        if not file_path.exists():                                          # Checks whether the raw log JSON file exists on disk yet
            return []                                                       # Returns an empty list safely instead of crashing if file is missing
        with open(file_path, "r", encoding="utf-8") as f:                   # Opens the JSON log file in UTF-8 read mode
            return json.load(f)                                             # Parses and returns the list of raw event dictionaries

    def fetch_signals(self, resource_filter: Optional[str] = None) -> List[IncidentSignal]: # Implements the required abstract method
        raw_items = self._load_raw_records()                                # Loads raw JSON events for this specific service
        signals: List[IncidentSignal] = []                                  # Initializes an empty list to store validated IncidentSignal objects
        for item in raw_items:                                              # Iterates through every raw log/metric entry in the file
            res_name = item.get("resource_name", "unknown")                 # Extracts the GCP resource name (DAG, job, table, or pipeline)
            if resource_filter and resource_filter.lower() not in res_name.lower(): # Checks if caller filtered for a specific resource
                continue                                                    # Skips events that do not match the requested resource filter
            signals.append(                                                 # Converts the raw dict into a validated IncidentSignal and appends it
                IncidentSignal(                                             # Instantiates our canonical Pydantic model (validates types)
                    timestamp=item["timestamp"],                            # Maps ISO-8601 timestamp string
                    service=self.service_key,                               # Stamps the exact connector key (e.g., 'bigquery_data_quality')
                    plane=self.plane,                                       # Stamps the signal plane ('alert', 'service_log', 'data_quality', 'cicd')
                    resource_name=res_name,                                 # Maps the impacted GCP asset name
                    severity=item.get("severity", "INFO"),                  # Maps severity level (defaults to 'INFO' if omitted)
                    metric_name=item.get("metric_name", "latency_ms"),      # Maps the primary numeric metric name
                    metric_value=float(item.get("metric_value", 0.0)),      # Casts observed numeric metric to float for math/ML engines
                    baseline_value=float(item.get("baseline_value", 0.0)),  # Casts baseline numeric metric to float for drift calculation
                    message=item.get("message", ""),                        # Maps unstructured error log, stack trace, or DQ alert message
                    metadata=item.get("metadata", {}),                      # Preserves extra tags (incident_id, job_id, query_id, etc.)
                )
            )
        return signals                                                      # Returns the normalized list of IncidentSignal objects


@ConnectorRegistry.register("dynatrace")                                    # Auto-registers this class under key 'dynatrace' in ConnectorRegistry
class DynatraceConnector(LocalJsonBasePlugin):                              # Connector plugin for Dynatrace Davis AI Problems & Alerts
    pass                                                                    # Inherits full normalization logic (override here for live DT API later)


@ConnectorRegistry.register("gcp_monitoring")                               # Auto-registers this class under key 'gcp_monitoring'
class GCPMonitoringConnector(LocalJsonBasePlugin):                          # Connector plugin for native GCP Cloud Monitoring Alert Policies
    pass                                                                    # Inherits normalization logic (override for `google-cloud-monitoring` later)


@ConnectorRegistry.register("cloud_composer")                               # Auto-registers this class under key 'cloud_composer'
class CloudComposerConnector(LocalJsonBasePlugin):                          # Connector plugin for GCP Cloud Composer (Airflow DAG/task logs)
    pass                                                                    # Inherits normalization logic (override for `google-cloud-logging` later)


@ConnectorRegistry.register("dataflow")                                     # Auto-registers this class under key 'dataflow'
class DataflowConnector(LocalJsonBasePlugin):                               # Connector plugin for GCP Cloud Dataflow (Apache Beam worker logs)
    pass                                                                    # Inherits normalization logic


@ConnectorRegistry.register("bigquery_data_quality")                        # Auto-registers this class under key 'bigquery_data_quality'
class BigQueryDQConnector(LocalJsonBasePlugin):                             # Connector plugin for BigQuery INFORMATION_SCHEMA, slots & dbt DQ checks
    pass                                                                    # Inherits normalization logic


@ConnectorRegistry.register("pubsub_streaming")                             # Auto-registers this class under key 'pubsub_streaming'
class PubSubConnector(LocalJsonBasePlugin):                                 # Connector plugin for Pub/Sub unacked message lag & dead-letter spikes
    pass                                                                    # Inherits normalization logic


@ConnectorRegistry.register("harness_terraform")                            # Auto-registers this class under key 'harness_terraform'
class HarnessTerraformConnector(LocalJsonBasePlugin):                       # Connector plugin for Harness CI/CD deployments & Terraform state applies
    pass                                                                    # Inherits normalization logic