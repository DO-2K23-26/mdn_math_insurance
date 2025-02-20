import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base_classification_model import BaseClassificationModel

class RandomForestModel(BaseClassificationModel):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X, y):
        self.model.fit(X, y)
        print("Modèle Random Forest entraîné avec succès.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, X):
        importances = self.model.feature_importances_
        df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        return df.sort_values(by='Importance', ascending=False)

    def get_visualization_data(self, X, y):
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        from sklearn.metrics import classification_report, log_loss, brier_score_loss
        report = classification_report(y, y_pred, output_dict=True)
        loss = log_loss(y, y_prob)
        brier = brier_score_loss(y, y_prob)
        importance_df = self.get_feature_importances(X)
        return {
            "report": report,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "logloss": loss,
            "brier_score": brier,
            "feature_importances": importance_df
        }
