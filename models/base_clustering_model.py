from .base_model import BaseModel
from abc import abstractmethod

class BaseClusteringModel(BaseModel):
    @abstractmethod
    def predict(self, X):
        """
        En clustering, la méthode predict() renvoie l'assignation des clusters.
        """
        pass

    def get_clustering_metrics(self, X):
        from sklearn.metrics import silhouette_score
        clusters = self.predict(X)
        score = silhouette_score(X, clusters)
        return {"silhouette_score": score}
