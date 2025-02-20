from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        """Entraîne le modèle avec les données d'entraînement."""
        pass

    @abstractmethod
    def predict(self, X):
        """Retourne les prédictions du modèle."""
        pass

    def get_evaluation_metrics(self, X, y):
        raise NotImplementedError("Cette méthode doit être implémentée dans la classe fille.")
