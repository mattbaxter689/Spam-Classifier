from mlflow.client import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from spam.data_configs.data_config import TestMetrics


class ChampionChallengerManager:
    """
    Helper class that controls the promotion logic for
    Champion/Challenger models. This is more as am example,
    and can be optimized far better than what is here
    """

    def __init__(
        self,
        challenger_metrics: TestMetrics,
        challenger_run_id: str,
        model_name: str = "SpamClassifier",
        threshold: float = 0.01,
    ) -> None:
        self.challenger_metrics = challenger_metrics
        self.challenger_run_id = challenger_run_id
        self.model_name = model_name
        self.threshold = threshold
        self.client = MlflowClient()

        # Register the challenger version in the registry
        self.challenger_version = self._get_latest_version()

        # Assign it to Staging stage
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=self.challenger_version.version,
            stage="Staging",
        )

    def _get_latest_version(self) -> ModelVersion:
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        return max(versions, key=lambda v: int(v.version))

    def _get_production_model(self) -> ModelVersion | None:
        """Fetch current Production (champion) version."""
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        for v in versions:
            if v.current_stage == "Production":
                return v
        return None

    def promote(self, precision_threshold: float = 0.7) -> None:
        """Promote challenger to Production if it beats current champion."""
        champion_version = self._get_production_model()

        if champion_version is None:
            # No champion yet → promote challenger
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=self.challenger_version.version,
                stage="Production",
            )
            print(
                f"No existing champion. Promoted version {self.challenger_version.version} as champion."
            )
            return

        # Fetch champion metrics
        champ_run_id = champion_version.run_id
        champ_metrics = self.client.get_run(champ_run_id).data.metrics
        champ_recall = champ_metrics.get("test_recall", 0.0)

        # Check challenger metrics
        if self.challenger_metrics.test_precision < precision_threshold:
            print(
                f"Challenger model does not meet min precision threshold "
                f"(precision {self.challenger_metrics.test_precision:.4f} < 0.90)"
            )
            return

        # Compare recall for promotion
        if self.challenger_metrics.test_recall > champ_recall + self.threshold:
            # Demote old champion to Archived (optional)
            self.client.transition_model_version_stage(
                name=self.model_name, version=champion_version.version, stage="Archived"
            )
            # Promote challenger
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=self.challenger_version.version,
                stage="Production",
            )
            print(
                f"Promoted version {self.challenger_version.version} as new champion "
                f"(recall {self.challenger_metrics.test_recall:.4f} > {champ_recall:.4f})"
            )
        else:
            print(
                f"Challenger recall {self.challenger_metrics.test_recall:.4f} did not beat champion "
                f"{champ_recall:.4f}"
            )
