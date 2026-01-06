import mlflow
from mlflow.client import MlflowClient
from mlflow.exceptions import RestException
from mlflow.entities.model_registry import ModelVersion
from torch import threshold_


class ChampionChallengerManager:
    def __init__(self, model_name: str = "spam_classifier") -> None:
        self.model_name = model_name
        self.client = MlflowClient()

    def _get_latest_version(self) -> ModelVersion:
        versions = self.client.get_latest_versions(self.model_name)
        if not versions:
            raise RuntimeError("No model versions found in registry")
        return versions[0]

    def promote(self, recall: float, threshold: float = 0.01) -> None:
        try:
            champion = self.client.get_model_version_by_alias(
                self.model_name, "champion"
            )
            champion_run_id: str | None = champion.run_id

            if champion_run_id is None:
                raise ValueError("Champion model has no run_id")

            champ_metrics = self.client.get_run(champion_run_id).data.metrics
            champ_recall = champ_metrics.get("test_recall", 0.0)
        except (RestException, ValueError, IndexError) as e:
            latest = self.client.get_latest_versions(self.model_name)[0]
            self.client.set_registered_model_alias(
                self.model_name, "champion", latest.version
            )
            print(f"No champion found. Promoted version {latest.version} as champion.")
            return

        latest = self._get_latest_version()
        if recall > champ_recall + threshold:
            self.client.set_registered_model_alias(
                self.model_name, "champion", latest.version
            )
            print(
                f"Promoted version {latest.version} as new champion "
                f"(recall {recall:.4f} > {champ_recall:.4f})"
            )
        else:
            print(
                f"Challenger recall {recall:.4f} did not beat champion "
                f"{champ_recall:.4f}"
            )
