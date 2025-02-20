import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from .base_regression_model import BaseRegressionModel

class LinearRegressionModel(BaseRegressionModel):
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)
        print("Modèle Linear Regression entraîné avec succès.")

    def predict(self, X):
        return self.model.predict(X)

    def get_visualization_data(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': self.model.coef_})
        return {
            "y_pred": y_pred,
            "mse": mse,
            "r2": r2,
            "coefficients": coef_df,
            "intercept": self.model.intercept_
        }
