import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .base_classification_model import BaseClassificationModel

class LogisticRegressionModel(BaseClassificationModel):
    def __init__(self, class_weight='balanced', random_state=42):
        self.model = LogisticRegression(class_weight=class_weight, random_state=random_state)
        self.scaler = None

    def train(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        print("Modèle Logistic Regression entraîné avec succès.")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_visualization_data(self, X, y):
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        from sklearn.metrics import classification_report, log_loss, brier_score_loss
        report = classification_report(y, y_pred, output_dict=True)
        loss = log_loss(y, y_prob)
        brier = brier_score_loss(y, y_prob)
        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': self.model.coef_[0]})
        coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
        return {
            "report": report,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "logloss": loss,
            "brier_score": brier,
            "coefficients": coef_df
        }
