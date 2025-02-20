import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .base_clustering_model import BaseClusteringModel

class KMeansModel(BaseClusteringModel):
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def train(self, X, y=None):
        self.model.fit(X)
        print("Modèle KMeans ajusté avec succès.")

    def predict(self, X):
        return self.model.predict(X)

    def get_visualization_data(self, X):
        clusters = self.predict(X)
        score = silhouette_score(X, clusters)
        centroids = self.model.cluster_centers_
        return {"clusters": clusters, "silhouette_score": score, "centroids": centroids}
