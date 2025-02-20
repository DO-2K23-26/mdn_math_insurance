from .base_model import BaseModel
from abc import abstractmethod

class BaseClassificationModel(BaseModel):
    @abstractmethod
    def predict_proba(self, X):
        """Retourne les probabilités pour la classe positive."""
        pass

    def get_classification_report(self, X, y):
        from sklearn.metrics import classification_report
        y_pred = self.predict(X)
        return classification_report(y, y_pred, output_dict=True)

    def get_classification_metrics(self, X, y):
        from sklearn.metrics import log_loss, brier_score_loss
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        report = self.get_classification_report(X, y)
        loss = log_loss(y, y_prob)
        brier = brier_score_loss(y, y_prob)
        return {"report": report, "logloss": loss, "brier_score": brier}
