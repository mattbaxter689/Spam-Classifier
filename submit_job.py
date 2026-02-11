import os
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# NOTE: This is an example if you have a config.json file from Azure. Otherwise,
# use a service principal
# config_path = Path(__file__).parent / "config.json"
# with open(config_path) as f:
#     ws_cfg = json.load(f)

compute = require_env("AML_COMPUTE")
environment_name = require_env("AML_ENVIRONMENT")
train_data = require_env("AML_TRAIN")
environment_image = require_env("AML_ENV_IMAGE")

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=require_env("subscription_id"),
    resource_group_name=require_env("resource_group"),
    workspace_name=require_env("workspace_name"),
)

# Define the job
job = command(
    code="./src",
    command="python -m spam.train --data_path ${{inputs.train_data}}",
    environment=f"{environment_name}@latest",
    compute=compute,
    inputs={
        "train_data": Input(
            type=AssetTypes.URI_FOLDER,
            path=train_data,
        )
    },
    # 👇 ENV VARS INSIDE THE CONTAINER
    environment_variables={"HF_TOKEN": require_env("HF_TOKEN")},
)

submitted_job = ml_client.jobs.create_or_update(job)
print(f"✅ Job submitted: {submitted_job.name}")

# Optional: stream logs immediately
ml_client.jobs.stream(str(submitted_job.name))
