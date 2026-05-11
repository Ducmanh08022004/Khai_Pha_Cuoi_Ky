from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


@dataclass
class ClusterDiagnostics:
    k_values: List[int]
    inertias: List[float]
    silhouettes: List[float]


def _dense_matrix(X):
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def choose_best_k(
    X,
    k_values: Sequence[int] = range(2, 11),
    random_state: int = 42,
) -> ClusterDiagnostics:
    dense = _dense_matrix(X)
    inertias: List[float] = []
    silhouettes: List[float] = []
    valid_k: List[int] = []

    for k in k_values:
        if dense.shape[0] <= k:
            continue
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = model.fit_predict(dense)
        valid_k.append(k)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(dense, labels)))

    return ClusterDiagnostics(k_values=valid_k, inertias=inertias, silhouettes=silhouettes)


def fit_kmeans_pp(X, n_clusters: int, random_state: int = 42) -> KMeans:
    dense = _dense_matrix(X)
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    model.fit(dense)
    return model


def assign_clusters(model: KMeans, X) -> np.ndarray:
    dense = _dense_matrix(X)
    return model.predict(dense)


def reduce_to_2d(X, random_state: int = 42) -> np.ndarray:
    dense = _dense_matrix(X)
    return PCA(n_components=2, random_state=random_state).fit_transform(dense)
