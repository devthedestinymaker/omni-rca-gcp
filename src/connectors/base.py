from abc import ABC, abstractmethod                                         # Imports tools to create Abstract Base Classes (enforces plugin contracts)
from typing import Any, Dict, List, Optional                                # Imports static type hints for cleaner, self-documenting code
from pydantic import BaseModel, Field                                       # Imports Pydantic for strict runtime data validation and JSON schemas
import yaml                                                                 # Imports PyYAML to parse our `configs/connectors.yaml` configuration file

# ==============================================================================
# WHAT: Canonical Signal Schema & Plug-and-Play Connector Architecture (TICKET-102)
# WHY:  Dynatrace alerts, GCP Monitoring alerts, Composer tracebacks, BigQuery
#       row-count drops, and Harness deployments all emit different raw JSON
#       schemas. Normalizing them into one unified `IncidentSignal` model ensures
#       Stats [02], Pandas [03], LightGBM [04], PyTorch [05], Vector DB [07], and
#       MCP Agent [08] work seamlessly across ANY service—and lets you plug in
#       new connectors later without modifying core code.
# HOW:  1. `IncidentSignal` (Pydantic BaseModel): Enforces strict typing and
#          validation on every ingested log or data-quality metric.
#       2. `BaseConnector` (Abstract Base Class): Guarantees every plugin class
#          implements `.fetch_signals()`.
#       3. `ConnectorRegistry` (Registry Decorator Pattern): Maps YAML keys to
#          Python classes and instantiates ONLY the connectors toggled ON.
# WHERE: Core ingestion foundation (`src/connectors/base.py`).
# ==============================================================================

class IncidentSignal(BaseModel):                                            # Defines a unified data model so logs & data-quality anomalies share 1 format
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")       # Required event timestamp used later to align events across GCP services
    service: str = Field(..., description="Connector key name")             # Identifies which connector produced this record (e.g., 'dataflow')
    plane: str = Field(..., description="alert|service_log|data_quality|cicd") # Separates alert triggers, service logs, data quality, and CI/CD events
    resource_name: str = Field(..., description="DAG, job, or BQ table")    # Exact GCP asset impacted (e.g., 'analytics.fact_orders' or 'dag_etl')
    severity: str = Field(default="INFO", description="Severity level")     # Standardized severity ('INFO', 'WARN', 'ERROR', 'CRITICAL')
    metric_name: str = Field(default="latency_ms", description="Metric")    # Name of the numeric telemetry or data-quality metric being measured
    metric_value: float = Field(default=0.0, description="Observed value")  # Actual observed value during the event (used by Stats & LightGBM)
    baseline_value: float = Field(default=0.0, description="Normal value")  # Historical normal value (lets us compute exact deviation/drift)
    message: str = Field(..., description="Error log or DQ summary")        # Unstructured text/stack trace (analyzed by PyTorch DL & Vector RAG)
    metadata: Dict[str, Any] = Field(default_factory=dict)                  # Flexible dictionary for extra service tags (job_id, query_id, etc.)


class BaseConnector(ABC):                                                   # Abstract parent class that every GCP/alert plugin must inherit from
    def __init__(self, service_key: str, config: Dict[str, Any]):           # Constructor receiving the YAML key and its configuration block
        self.service_key = service_key                                      # Stores the unique service identifier (e.g., 'cloud_composer')
        self.config = config                                                # Stores the full dictionary loaded from `connectors.yaml`
        self.enabled: bool = config.get("enabled", False)                   # Reads whether this connector is enabled by default (defaults to False)
        self.plane: str = config.get("plane", "service_log")                # Stores the operational plane ('alert', 'service_log', etc.)
        self.display_name: str = config.get("display_name", service_key)    # Stores the friendly UI display name

    @abstractmethod                                                         # Decorator forcing every child connector class to implement this method
    def fetch_signals(self, resource_filter: Optional[str] = None) -> List[IncidentSignal]: # Contract method returning standardized IncidentSignal objects
        raise NotImplementedError("Subclasses must implement fetch_signals()") # Safety error if a developer forgets to override this method


class ConnectorRegistry:                                                    # Central plugin manager (Registry Pattern) to load connectors dynamically
    _REGISTRY: Dict[str, type[BaseConnector]] = {}                          # Class-level dictionary mapping service keys ('dataflow') to Python classes

    @classmethod                                                            # Allows calling `.register()` directly on the class without instantiating it
    def register(cls, service_key: str):                                    # Decorator factory accepting the YAML key name to register a plugin under
        def decorator(connector_cls: type[BaseConnector]) -> type[BaseConnector]: # Inner decorator receiving the actual connector class being defined
            cls._REGISTRY[service_key] = connector_cls                      # Saves the class reference into `_REGISTRY` for instant lookup later
            return connector_cls                                            # Returns the unmodified class so normal Python behavior is preserved
        return decorator                                                    # Returns the configured decorator function

    @classmethod                                                            # Class method to inspect available connectors before instantiating them
    def list_available_configs(cls, config_path: str = "configs/connectors.yaml") -> Dict[str, Dict[str, Any]]: # Reads YAML config file
        with open(config_path, "r", encoding="utf-8") as f:                 # Opens `connectors.yaml` safely in UTF-8 read mode
            return yaml.safe_load(f).get("connectors", {})                  # Parses YAML safely and returns the `connectors` dictionary

    @classmethod                                                            # Class method that builds ONLY the connectors toggled ON by the user
    def load_connectors(                                                    # Factory method used by the UI and MCP Server at runtime
        cls,                                                                # Reference to the ConnectorRegistry class
        config_path: str = "configs/connectors.yaml",                       # Path to the YAML configuration file
        selected_services: Optional[List[str]] = None,                      # Optional list of service keys checked by the engineer in the UI
    ) -> Dict[str, BaseConnector]:                                          # Returns a dictionary of active, ready-to-use connector instances
        all_configs = cls.list_available_configs(config_path)               # Loads all connector definitions from `connectors.yaml`
        active_instances: Dict[str, BaseConnector] = {}                     # Initializes an empty dictionary to hold active connector objects

        for key, cfg in all_configs.items():                                # Loops through every configured connector in `connectors.yaml`
            should_enable = (                                               # Decides whether to activate this specific connector
                (key in selected_services)                                  # If UI checkboxes were passed, check if `key` is in that list
                if selected_services is not None                            # Condition checking whether a runtime UI selection exists
                else cfg.get("enabled", False)                              # Otherwise fall back to the default `enabled` flag in YAML
            )
            if should_enable and key in cls._REGISTRY:                      # Ensures the connector is both enabled AND registered in Python
                active_instances[key] = cls._REGISTRY[key](service_key=key, config=cfg) # Instantiates the plugin class and stores it by key

        return active_instances                                             # Returns only the active connectors so disabled services use zero resources