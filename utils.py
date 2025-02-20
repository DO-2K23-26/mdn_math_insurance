def display_visualization_data(viz_data: dict, model_name: str):
    """
    Affiche dans la console les données de visualisation retournées par get_visualization_data.
    """
    print(f"\n==== Rapport de Visualisation pour {model_name} ====\n")

    report = viz_data.get("report", {})
    if report:
        print("Classification Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"\nLabel: {label}")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            else:
                print(f"{label}: {metrics}")
    else:
        print("Aucun rapport de classification disponible.")

    logloss = viz_data.get("logloss", None)
    brier = viz_data.get("brier_score", None)
    if logloss is not None:
        print(f"\nLogloss: {logloss:.4f}")
    if brier is not None:
        print(f"Brier Score: {brier:.4f}")

    if "coefficients" in viz_data:
        print("\nCoefficients:")
        print(viz_data["coefficients"].to_string(index=False))
    elif "feature_importances" in viz_data:
        print("\nFeature Importances:")
        print(viz_data["feature_importances"].to_string(index=False))

    print("\n==== Fin du Rapport ====\n")


def display_model_metrics(viz_data: dict, model_name: str):
    """
    Affiche dans la console les métriques contenues dans le dictionnaire viz_data,
    en adaptant l'affichage selon le type de modèle (classification, régression, clustering).

    Parameters:
        viz_data (dict): Dictionnaire retourné par la méthode d'évaluation du modèle.
        model_name (str): Nom du modèle (pour l'affichage contextuel).
    """
    print(f"\n==== Metrics for {model_name} ====\n")

    # Cas Classification (présence d'un 'report' et des métriques de logloss / Brier score)
    if "report" in viz_data:
        print("Classification Report:")
        for label, metrics in viz_data["report"].items():
            # Affichage formaté pour les métriques (pour les labels individuels)
            if isinstance(metrics, dict):
                print(f"  {label}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value}")
            else:
                print(f"  {label}: {metrics}")
        print(f"\nLogloss: {viz_data.get('logloss', 'N/A'):.4f}")
        print(f"Brier Score: {viz_data.get('brier_score', 'N/A'):.4f}")

    # Cas Régression (présence de 'mse' et 'r2')
    elif "mse" in viz_data:
        print("Regression Metrics:")
        print(f"  Mean Squared Error (MSE): {viz_data['mse']:.4f}")
        print(f"  R² Score: {viz_data['r2']:.4f}")
        print("\nCoefficients:")
        print(viz_data["coefficients"].to_string(index=False))
        # Vous pouvez ajouter l'intercept si souhaité :
        if "intercept" in viz_data:
            print(f"\nIntercept: {viz_data['intercept']:.4f}")

    # Cas Clustering (présence de 'silhouette_score')
    elif "silhouette_score" in viz_data:
        print("Clustering Metrics:")
        print(f"  Silhouette Score: {viz_data['silhouette_score']:.4f}")
        print("\nCentroids:")
        print(viz_data["centroids"])

    else:
        print("Aucune métrique reconnue n'est disponible dans viz_data.")

    print("\n==== End Metrics ====\n")
