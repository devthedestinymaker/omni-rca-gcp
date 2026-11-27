# ==============================================================================
# WHAT: Cross-Service & Cross-Plane Incident Correlator [Topic 03: Data Analysis]
# WHY:  In GCP, a single failure ripples across planes: a Harness deploy or
#       BigQuery 0-row load happens first, causing a Dataflow OOM or dbt failure
#       minutes later, which finally triggers a Dynatrace or GCP Alert. We need
#       Pandas time-window alignment to reconstruct that exact causal chain.
# HOW:  Ingests `IncidentSignal` lists from all active connectors, builds a
#       time-indexed Pandas DataFrame, calculates baseline percentage drift and
#       time deltas relative to the alert trigger, and extracts correlated windows.
# WHERE: `src/intelligence/log_correlator.py` (used by the MCP correlation tool,
#        LightGBM feature builder, and Streamlit timeline view).
# EXAMPLE:
#   from src.connectors import ConnectorRegistry
#   from src.intelligence.log_correlator import MultiServiceLogCorrelator
#   conns = ConnectorRegistry.load_connectors()
#   correlator = MultiServiceLogCorrelator.from_connectors(conns)
#   timeline_df = correlator.get_incident_window("2026-11-27T02:15:00Z", window_minutes=30)
# ==============================================================================

from typing import Dict, List, Optional                                     # Imports static type hints for dictionaries, lists, and optionals
import numpy as np                                                          # Imports NumPy for safe vectorized division without zero-errors
import pandas as pd                                                         # Imports Pandas for time-series alignment, filtering, and grouping
from src.connectors.base import BaseConnector, IncidentSignal               # Imports base connector interface and canonical signal schema


class MultiServiceLogCorrelator:                                            # Aligns telemetry across Alert, Service-Log, Data-Quality & CI/CD planes
    def __init__(self, signals: List[IncidentSignal]):                      # Initializes correlator with a combined list of IncidentSignal objects
        self.signals = signals                                              # Stores raw Pydantic signal objects for reference
        self.df = self._build_dataframe(signals)                            # Immediately converts signals into a clean, sorted Pandas DataFrame

    @classmethod                                                            # Factory constructor to pull signals directly from active connectors
    def from_connectors(                                                    # Convenience builder taking the dict returned by `load_connectors()`
        cls, active_connectors: Dict[str, BaseConnector], resource_filter: Optional[str] = None
    ) -> "MultiServiceLogCorrelator":
        all_signals: List[IncidentSignal] = []                              # Initializes master list to aggregate signals across all enabled tools
        for connector in active_connectors.values():                        # Iterates through each active plugin (Dynatrace, Composer, BQ, etc.)
            all_signals.extend(connector.fetch_signals(resource_filter=resource_filter)) # Fetches & appends normalized signals from each connector
        return cls(all_signals)                                             # Returns an initialized MultiServiceLogCorrelator instance

    def _build_dataframe(self, signals: List[IncidentSignal]) -> pd.DataFrame: # Converts Pydantic models into an enriched time-series DataFrame
        if not signals:                                                     # Guards against empty signal lists when no logs match
            return pd.DataFrame()                                           # Returns an empty DataFrame cleanly
        records = [s.model_dump() for s in signals]                         # Serializes each Pydantic IncidentSignal into a standard Python dict
        df = pd.DataFrame(records)                                          # Builds a Pandas DataFrame from the list of dictionaries
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)         # Parses ISO-8601 strings into timezone-aware UTC datetime objects
        df["incident_id"] = df["metadata"].apply(                           # Extracts `incident_id` tag from nested metadata dict into a column
            lambda m: m.get("incident_id", "UNKNOWN") if isinstance(m, dict) else "UNKNOWN"
        )
        safe_base = df["baseline_value"].replace(0, np.nan)                 # Replaces 0 baselines with NaN to avoid ZeroDivisionError
        df["pct_deviation"] = (                                             # Computes percentage change from historical baseline
            ((df["metric_value"] - df["baseline_value"]) / safe_base) * 100.0
        ).fillna(0.0).round(2)                                              # Fills NaN (when baseline was 0) with 0.0 and rounds to 2 decimals
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True) # Sorts chronologically so root causes appear before symptoms
        return df                                                           # Returns the enriched, chronologically ordered DataFrame

    def get_incident_window(                                                # Extracts all cross-service events within +/- N minutes of an alert
        self, trigger_timestamp: str, window_minutes: int = 30
    ) -> pd.DataFrame:
        if self.df.empty:                                                   # Checks if DataFrame has any records
            return self.df                                                  # Returns empty DataFrame if no data exists
        center_ts = pd.to_datetime(trigger_timestamp, utc=True)             # Parses the target alert timestamp into UTC datetime
        delta_mins = (self.df["timestamp"] - center_ts).dt.total_seconds() / 60.0 # Calculates signed minutes before (-) or after (+) the alert
        window_df = self.df.copy()                                          # Copies DataFrame to avoid mutating internal state
        window_df["delta_mins"] = delta_mins.round(2)                       # Adds `delta_mins` column (-4.92 means 4.92 mins BEFORE alert)
        mask = window_df["delta_mins"].abs() <= window_minutes              # Filters rows falling inside [-window_minutes, +window_minutes]
        return window_df.loc[mask, [                                        # Returns clean, focused diagnostic columns for the incident window
            "timestamp", "delta_mins", "incident_id", "plane", "service",
            "resource_name", "severity", "metric_name", "metric_value",
            "baseline_value", "pct_deviation", "message"
        ]].reset_index(drop=True)                                           # Resets index cleanly for display and downstream ML feature extraction

    def summarize_by_plane(self, window_df: pd.DataFrame) -> Dict[str, int]: # Counts correlated signals per operational plane
        if window_df.empty:                                                 # Guards against empty window DataFrames
            return {}                                                       # Returns empty summary dictionary
        return window_df["plane"].value_counts().to_dict()                  # Counts events per plane ('alert', 'service_log', 'data_quality')