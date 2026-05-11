from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover - optional dependency
    SMOTE = None
    ImbPipeline = None


@dataclass
class ClassificationResults:
    report: Dict[str, Any]
    confusion: np.ndarray
    roc_auc: float | None


def train_random_forest(
    X_train,
    y_train,
    use_smote: bool = True,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = None,
) -> Any:
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    if use_smote and SMOTE is not None and ImbPipeline is not None:
        model = ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=random_state)),
                ("rf", forest),
            ]
        )
    else:
        model = forest

    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model: Any, X_test, y_test) -> ClassificationResults:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_test, y_pred)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_score))
        except Exception:
            roc_auc = None

    return ClassificationResults(report=report, confusion=confusion, roc_auc=roc_auc)


def get_feature_importance(model: Any, feature_names: Sequence[str]) -> Dict[str, float]:
    estimator = model
    if hasattr(model, "named_steps") and "rf" in model.named_steps:
        estimator = model.named_steps["rf"]
    if not hasattr(estimator, "feature_importances_"):
        return {}

    importances = estimator.feature_importances_
    paired = zip(feature_names, importances)
    return dict(sorted(((name, float(score)) for name, score in paired), key=lambda item: item[1], reverse=True))
