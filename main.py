from data_manager import DataManager
from utils import display_visualization_data
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.linear_regression_model import LinearRegressionModel
from models.kmeans_model import KMeansModel

def main():
    data_path = "data/insurance.csv"

    # Chargement et prétraitement des données
    data_manager = DataManager(data_path)
    data_manager.load_data()

    # ====================================
    # Tâche de Classification
    # ====================================
    print("\n=== Tâche de Classification ===")
    features_clf, target_clf = data_manager.preprocess_classification()
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = data_manager.split_data(features_clf, target_clf, test_size=0.2, random_state=42)

    classification_models = [
        ("Logistic Regression", LogisticRegressionModel()),
        ("Random Forest", RandomForestModel(n_estimators=100, random_state=42))
    ]

    for name, model in classification_models:
        print(f"\n==== {name} (Classification) ====")
        model.train(X_train_clf, y_train_clf)
        viz_data = model.get_visualization_data(X_test_clf, y_test_clf)
        display_visualization_data(viz_data, name)

    # ====================================
    # Tâche de Régression
    # ====================================
    print("\n=== Tâche de Régression ===")
    features_reg, target_reg = data_manager.preprocess_regression()
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = data_manager.split_data(features_reg, target_reg, test_size=0.2, random_state=42)

    reg_model = LinearRegressionModel()
    print("\n==== Linear Regression (Régression) ====")
    reg_model.train(X_train_reg, y_train_reg)
    viz_data_reg = reg_model.get_visualization_data(X_test_reg, y_test_reg)
    display_visualization_data(viz_data_reg, "Linear Regression")

    # ====================================
    # Tâche de Clustering
    # ====================================
    print("\n=== Tâche de Clustering ===")
    features_clust = data_manager.preprocess_clustering()
    clust_model = KMeansModel(n_clusters=3, random_state=42)
    print("\n==== KMeans Clustering (Clustering) ====")
    clust_model.train(features_clust, None)  # y n'est pas utilisé en clustering
    viz_data_clust = clust_model.get_visualization_data(features_clust)
    display_visualization_data(viz_data_clust, "KMeans Clustering")


if __name__ == '__main__':
    main()
