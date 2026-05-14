from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .association_rules import RuleMiningConfig, build_transactions, mine_association_rules
from .classification import ClassificationResults, evaluate_classifier, get_feature_importance, train_random_forest
from .clustering import ClusterDiagnostics, assign_clusters, choose_best_k, fit_kmeans_pp
from .features import FeatureBundle, build_feature_bundle, feature_names, transform_feature_bundle
from .preprocessing import preprocess_dataframe
from .recommendation import AlertResult, build_alert, combine_risk_scores


@dataclass
class FraudDetectionArtifacts:
    feature_bundle: FeatureBundle | None = None
    classifier: Any | None = None
    cluster_model: Any | None = None
    cluster_risk_map: Dict[int, float] = field(default_factory=dict)
    rules: pd.DataFrame = field(default_factory=pd.DataFrame)
    cluster_diagnostics: ClusterDiagnostics | None = None


@dataclass
class TrainingSummary:
    evaluation: ClassificationResults | None = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    cluster_diagnostics: ClusterDiagnostics | None = None
    rules: pd.DataFrame = field(default_factory=pd.DataFrame)


class FraudDetectionPipeline:
    def __init__(
        self,
        text_col: str = "text",
        label_col: str = "label",
        n_clusters: int = 5,
        random_state: int = 42,
        n_estimators: int = 300,
        max_features: int = 5000,
        min_support: float = 0.15,
        skip_rules: bool = False,
    ) -> None:
        self.text_col = text_col
        self.label_col = label_col
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_support = min_support
        self.skip_rules = skip_rules
        self.artifacts = FraudDetectionArtifacts()

    def _coerce_labels(self, y: pd.Series) -> pd.Series:
        values = y.astype(str).str.lower().str.strip()
        mapping = {
            # Fraud/Spam indicators
            "spam": 1,
            "fraud": 1,
            "scam": 1,
            "phishing": 1,
            "lừa đảo": 1,
            "1": 1,
            # Legitimate/Ham indicators
            "ham": 0,
            "legit": 0,
            "legitimate": 0,
            "safe": 0,
            "hợp lệ": 0,
            "0": 0,
        }
        return values.map(mapping).fillna(values.astype(float, errors="ignore")).astype(int)

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, use_smote: bool = True) -> TrainingSummary:
        if self.text_col not in df.columns:
            raise KeyError(f"Missing text column: {self.text_col}")
        if self.label_col not in df.columns:
            raise KeyError(f"Missing label column: {self.label_col}")

        print("[info] Stage 2/6: Preprocessing text data...")
        prepared = preprocess_dataframe(df, text_col=self.text_col, language="en")
        y = self._coerce_labels(prepared[self.label_col])
        
        print("[info] Stage 3/6: Splitting into train/test sets...")
        train_df, test_df, y_train, y_test = train_test_split(
            prepared,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        print("[info] Stage 4/6: Building feature vectors...")
        bundle, X_train = build_feature_bundle(train_df, text_col="clean_text", max_features=self.max_features)
        X_test = transform_feature_bundle(bundle, test_df, text_col="clean_text")

        print("[info] Stage 5/6: Training Random Forest classifier...")
        classifier = train_random_forest(X_train, y_train, use_smote=use_smote, random_state=self.random_state, n_estimators=self.n_estimators)
        evaluation = evaluate_classifier(classifier, X_test, y_test)

        feature_importance = get_feature_importance(classifier, feature_names(bundle))

        cluster_source = train_df[y_train == 1]
        if cluster_source.empty:
            cluster_source = train_df
        cluster_matrix = transform_feature_bundle(bundle, cluster_source, text_col="clean_text")
        k = min(self.n_clusters, max(2, cluster_source.shape[0] - 1))
        cluster_diagnostics = choose_best_k(cluster_matrix, k_values=range(2, max(3, k + 1)), random_state=self.random_state)
        cluster_model = fit_kmeans_pp(cluster_matrix, n_clusters=k, random_state=self.random_state)

        cluster_labels = assign_clusters(cluster_model, cluster_matrix)
        cluster_risk_map: Dict[int, float] = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_rows = cluster_source.iloc[np.where(cluster_labels == cluster_id)[0]]
            if cluster_rows.empty:
                cluster_risk_map[int(cluster_id)] = 0.0
            else:
                cluster_risk_map[int(cluster_id)] = float(y_train.loc[cluster_rows.index].mean())

        print("[info] Stage 6/6: Mining association rules...")
        rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
        
        if not self.skip_rules:
            transactions = build_transactions(
                train_df[y_train == 1] if (y_train == 1).any() else train_df,
                text_col="clean_text",
                extra_item_columns=[
                    column
                    for column in ["has_url", "has_shorturl", "has_phone", "has_money", "bank_mention"]
                    if column in train_df.columns
                ],
            )
            try:
                config = RuleMiningConfig(min_support=self.min_support)
                rules = mine_association_rules(transactions, config)
            except ImportError:
                pass
        else:
            print("[info] Skipping association rule mining")

        print("[info] Finalizing results...")
        self.artifacts = FraudDetectionArtifacts(
            feature_bundle=bundle,
            classifier=classifier,
            cluster_model=cluster_model,
            cluster_risk_map=cluster_risk_map,
            rules=rules,
            cluster_diagnostics=cluster_diagnostics,
        )

        return TrainingSummary(
            evaluation=evaluation,
            feature_importance=feature_importance,
            cluster_diagnostics=cluster_diagnostics,
            rules=rules,
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.artifacts.feature_bundle is None or self.artifacts.classifier is None:
            raise RuntimeError("Pipeline has not been fitted yet")

        frame = pd.DataFrame({self.text_col: list(texts)})
        prepared = preprocess_dataframe(frame, text_col=self.text_col)
        matrix = transform_feature_bundle(self.artifacts.feature_bundle, prepared, text_col="clean_text")
        return self.artifacts.classifier.predict_proba(matrix)[:, 1]

    def predict(self, texts: Sequence[str]) -> pd.DataFrame:
        if self.artifacts.feature_bundle is None or self.artifacts.classifier is None:
            raise RuntimeError("Pipeline has not been fitted yet")

        frame = pd.DataFrame({self.text_col: list(texts)})
        prepared = preprocess_dataframe(frame, text_col=self.text_col)
        matrix = transform_feature_bundle(self.artifacts.feature_bundle, prepared, text_col="clean_text")
        proba = self.artifacts.classifier.predict_proba(matrix)[:, 1]
        cluster_ids = assign_clusters(self.artifacts.cluster_model, matrix) if self.artifacts.cluster_model is not None else np.zeros(len(frame), dtype=int)

        alerts = []
        for index, probability in enumerate(proba):
            cluster_id = int(cluster_ids[index]) if len(cluster_ids) else 0
            cluster_risk = self.artifacts.cluster_risk_map.get(cluster_id, 0.0)
            rule_hits = 0
            if not self.artifacts.rules.empty:
                tokens = set(prepared.iloc[index]["clean_text"].split())
                for _, rule in self.artifacts.rules.iterrows():
                    antecedents = set(rule["antecedents"])
                    if antecedents and antecedents.issubset(tokens):
                        rule_hits += 1
            alerts.append(build_alert(probability, cluster_risk=cluster_risk, rule_hits=rule_hits))

        return pd.DataFrame(
            {
                "text": list(texts),
                "fraud_probability": proba,
                "cluster_id": cluster_ids,
                "risk_score": [alert.score for alert in alerts],
                "severity": [alert.severity for alert in alerts],
            }
        )

    def score_message(self, text: str) -> AlertResult:
        prediction = self.predict([text]).iloc[0]
        reasons = [f"Cluster {int(prediction['cluster_id'])}"]
        return build_alert(
            fraud_probability=float(prediction["fraud_probability"]),
            cluster_risk=float(prediction["risk_score"]),
            rule_hits=0,
            reasons=reasons,
        )

    def save_model(self, model_path: str | Path) -> None:
        """Save the trained model artifacts to disk using joblib."""
        if self.artifacts.classifier is None:
            raise RuntimeError("Pipeline has not been fitted yet. Train model first.")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.artifacts, model_path)
        print(f"[info] Model saved to {model_path}")

    @classmethod
    def load_model(cls, model_path: str | Path) -> FraudDetectionPipeline:
        """Load a previously trained model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        artifacts = joblib.load(model_path)
        
        # Create a new pipeline instance and restore artifacts
        pipeline = cls()
        pipeline.artifacts = artifacts
        print(f"[info] Model loaded from {model_path}")
        return pipeline
