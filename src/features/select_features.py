from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

class FeatureSelection():
    def __init__(self, X: np.ndarray):
        self.features = X

    def variance_threshold(self):
        # drop zero/near-zero variance features
        self.vt = VarianceThreshold(threshold=0.01)
        self.features = self.vt.fit_transform(self.features)

    def silhouette_ablation(self, k: int = 4) -> dict[int, float]:
        # drop each feature and measure silhouette degradation
        base_score = silhouette_score(
            self.features, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(self.features)
        )
        drops = {}
        for i in range(self.features.shape[1]):
            feature_drop = np.delete(self.features, i, axis=1)
            score = silhouette_score(
                feature_drop, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(feature_drop)
            )
            drops[i] = base_score - score
        return drops
    