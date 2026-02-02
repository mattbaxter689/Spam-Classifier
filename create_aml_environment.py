from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
import os
import json
from pathlib import Path

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


config_path = Path(__file__).parent / "config.json"
with open(config_path) as f:
    ws_cfg = json.load(f)

environment_name = require_env("AML_ENVIRONMENT")
environment_image = require_env("AML_ENV_IMAGE")

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=ws_cfg["subscription_id"],
    resource_group_name=ws_cfg["resource_group"],
    workspace_name=ws_cfg["workspace_name"],
)

# After connecting to the client, we need to create the environment
job_env = Environment(
    name=environment_name,
    description="Customer environment for training spam classifier",
    tags={"project": "spam-classifier"},
    # conda_file=os.path.join("environment", "conda.yaml"),
    # image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    image=environment_image
)

# Register environment (create_or_update is idempotent)
job_env = ml_client.environments.create_or_update(job_env)
print(f"✅ Environment '{job_env.name}' registered with version {job_env.version}")
