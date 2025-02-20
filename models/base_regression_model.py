from .base_model import BaseModel

class BaseRegressionModel(BaseModel):
    def get_regression_metrics(self, X, y):
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return {"mse": mse, "r2": r2}
