
configs/connectors.yaml

src/connectors/base.py

Why We Built These Two Files First
Before writing any connector logic, we needed a contract (IncidentSignal + BaseConnector) and a switchboard (ConnectorRegistry + connectors.yaml).

When you run ConnectorRegistry.list_available_configs(), Python opens configs/connectors.yaml, parses the YAML text into a Python dictionary, and returns all 7 configured tool names (dynatrace, gcp_monitoring, cloud_composer, dataflow, bigquery_data_quality, pubsub_streaming, harness_terraform).

This proves your Python environment (PyYAML and Pydantic) is working, your folder structure is recognized as a proper Python package, and your configuration switchboard is readable.

--Output

That printed all 7 connector keys (['dynatrace', 'gcp_monitoring', 'cloud_composer', 'dataflow', 'bigquery_data_quality', 'pubsub_streaming', 'harness_terraform']) cleanly—TICKET-102 is officially verified and complete!   

Why What You Just Ran MattersConnectorRegistry read configs/connectors.yaml without a single hardcoded if/else statement.Every alert source (dynatrace, gcp_monitoring), service log source (cloud_composer, dataflow), data-quality source (bigquery_data_quality, pubsub_streaming), and CI/CD source (harness_terraform) is now registered in your configuration switchboard.