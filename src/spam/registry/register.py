from mlflow.client import MlflowClient
from mlflow.exceptions import RestException
from mlflow.entities.model_registry import ModelVersion
from spam.data_configs.data_config import TestMetrics


class ChampionChallengerManager:
    def __init__(
        self, challenger_metrics: TestMetrics, model_name: str = "SpamClassifier"
    ) -> None:
        self.challenger_metrics = challenger_metrics
        self.model_name = model_name
        self.client = MlflowClient()

        self._set_challenger_model()

    def _set_challenger_model(self) -> None:
        latest_version = self._get_latest_version()
        self.client.set_registered_model_alias(
            self.model_name, "challenger", latest_version.version
        )

    def _get_latest_version(self) -> ModelVersion:
        versions = self.client.get_latest_versions(self.model_name)
        if not versions:
            raise RuntimeError("No model versions found in registry")
        return versions[0]

    def promote(self, threshold: float = 0.01) -> None:
        try:
            champion = self.client.get_model_version_by_alias(
                self.model_name, "champion"
            )
            champion_run_id: str | None = champion.run_id

            if champion_run_id is None:
                raise ValueError("Champion model has no run_id")

            champ_metrics = self.client.get_run(champion_run_id).data.metrics
            champ_recall = champ_metrics.get("test_recall", 0.0)
        except (RestException, ValueError, IndexError):
            latest = self.client.get_latest_versions(self.model_name)[0]
            self.client.set_registered_model_alias(
                self.model_name, "champion", latest.version
            )
            print(f"No champion found. Promoted version {latest.version} as champion.")
            return

        latest = self._get_latest_version()
        if self.challenger_metrics.test_precision < 0.90:
            print(
                f"Challenger model does not meet min precision threshold"
                f"(precision {self.challenger_metrics.test_precision:.4f} < 0.90"
            )

        if self.challenger_metrics.test_recall > champ_recall + threshold:
            self.client.set_registered_model_alias(
                self.model_name, "champion", latest.version
            )
            print(
                f"Promoted version {latest.version} as new champion "
                f"(recall {self.challenger_metrics.test_recall:.4f} > {champ_recall:.4f})"
            )
        else:
            print(
                f"Challenger recall {self.challenger_metrics.test_recall:.4f} did not beat champion "
                f"{champ_recall:.4f}"
            )
