from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

MANUAL_FEATURE_COLUMNS = [
    "has_url",
    "has_shorturl",
    "has_phone",
    "has_money",
    "has_bank_account",
    "bank_mention",
    "fraud_keyword_count",
    "exclamation_count",
    "capslock_ratio",
    "msg_length",
]


@dataclass
class FeatureBundle:
    matrix: csr_matrix
    text_vectorizer: TfidfVectorizer
    manual_feature_columns: List[str]
    scaler: MinMaxScaler | None = None


def build_text_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )


def _manual_feature_frame(df: pd.DataFrame, feature_columns: Sequence[str] = MANUAL_FEATURE_COLUMNS) -> pd.DataFrame:
    frame = df.reindex(columns=feature_columns).fillna(0.0)
    return frame.astype(float)


def build_feature_bundle(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    feature_columns: Sequence[str] = MANUAL_FEATURE_COLUMNS,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[FeatureBundle, csr_matrix]:
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    texts = df[text_col].fillna("").astype(str)
    vectorizer = build_text_vectorizer(max_features=max_features, ngram_range=ngram_range)
    text_matrix = vectorizer.fit_transform(texts)
    
    manual_frame = _manual_feature_frame(df, feature_columns=feature_columns)
    scaler = MinMaxScaler()
    manual_matrix_scaled = scaler.fit_transform(manual_frame)
    # Giảm trọng số manual features xuống 10 lần để không lấn át TF-IDF
    manual_matrix_scaled = manual_matrix_scaled * 0.1
    manual_matrix = csr_matrix(manual_matrix_scaled)
    
    matrix = hstack([text_matrix, manual_matrix], format="csr")
    bundle = FeatureBundle(
        matrix=matrix,
        text_vectorizer=vectorizer,
        manual_feature_columns=list(feature_columns),
        scaler=scaler,
    )
    return bundle, matrix


def transform_feature_bundle(bundle: FeatureBundle, df: pd.DataFrame, text_col: str = "clean_text") -> csr_matrix:
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    texts = df[text_col].fillna("").astype(str)
    text_matrix = bundle.text_vectorizer.transform(texts)
    
    manual_frame = _manual_feature_frame(df, feature_columns=bundle.manual_feature_columns)
    if bundle.scaler is not None:
        manual_matrix_scaled = bundle.scaler.transform(manual_frame)
        manual_matrix_scaled = manual_matrix_scaled * 0.1
    else:
        manual_matrix_scaled = manual_frame.to_numpy()
        
    manual_matrix = csr_matrix(manual_matrix_scaled)
    return hstack([text_matrix, manual_matrix], format="csr")


def feature_names(bundle: FeatureBundle) -> List[str]:
    text_features = list(bundle.text_vectorizer.get_feature_names_out())
    return text_features + list(bundle.manual_feature_columns)
