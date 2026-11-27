# ==============================================================================
# WHAT: Package Initializer for `src.connectors`
# WHY:  In Python, `@ConnectorRegistry.register(...)` decorators only execute when
#       the module defining those classes is imported. Importing `gcp_plugins`
#       here guarantees all 7 connectors self-register automatically.
# HOW:  Re-exports `BaseConnector`, `ConnectorRegistry`, and `IncidentSignal`.
# WHERE: `src/connectors/__init__.py`
# ==============================================================================

from src.connectors.base import BaseConnector, ConnectorRegistry, IncidentSignal # Exposes core base classes at package level
import src.connectors.gcp_plugins  # noqa: F401                               # Imports plugin definitions so @register decorators execute immediately

__all__ = ["BaseConnector", "ConnectorRegistry", "IncidentSignal"]          # Defines explicit public exports for clean imports