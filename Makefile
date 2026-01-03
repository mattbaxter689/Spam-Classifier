IMAGE=ghcr.io/mlflow/mlflow:v3.8.1
CONTAINER_NAME=mlflow

DB_PATH=$(PWD)/mlflow.db
ARTIFACTS_PATH=$(PWD)/mlflow-artifacts

.PHONY: up down logs clean status

up:
	@mkdir -p $(ARTIFACTS_PATH)
	@touch $(DB_PATH)
	podman run -d \
	  --name $(CONTAINER_NAME) \
	  -p 5000:5000 \
	  -v $(DB_PATH):/mlflow/mlflow.db:Z \
	  -v $(ARTIFACTS_PATH):/mlflow/artifacts:Z \
	  $(IMAGE) \
	  mlflow server \
	    --backend-store-uri sqlite:///mlflow/mlflow.db \
	    --default-artifact-root /mlflow/artifacts \
	    --host 0.0.0.0 \
	    --port 5000

down:
	podman stop $(CONTAINER_NAME) || true
	podman rm $(CONTAINER_NAME) || true

logs:
	podman logs -f $(CONTAINER_NAME)

status:
	podman ps -a | grep $(CONTAINER_NAME) || true

clean: down
	rm -f $(DB_PATH)
	rm -rf $(ARTIFACTS_PATH)

