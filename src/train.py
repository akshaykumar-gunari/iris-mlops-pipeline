import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib

import warnings
warnings.filterwarnings('ignore')

def train_and_log():
    # Load data
    df = pd.read_csv('data/iris.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }

    best_model_name = None
    best_f1 = 0
    best_run_id = None
    best_artifact_path = None # New variable to store the correct artifact path

    for model_name, model in models.items():
        # Using the corrected log_model with name and input_example
        with mlflow.start_run(run_name=model_name):
            # Train
            model.fit(X_train, y_train)

            # Predict
            preds = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')

            # Log params
            if model_name == "LogisticRegression":
                mlflow.log_param("max_iter", model.max_iter)
            elif model_name == "RandomForest":
                mlflow.log_param("n_estimators", model.n_estimators)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Log model with name and input_example
            # This is the correct way to log the model
            logged_model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="IrisClassifier",
                input_example=X_train.iloc[0:1]
            )

            print(f"{model_name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

            # Track best model by f1 score
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_run_id = mlflow.active_run().info.run_id
                # The artifact path is "my-model", as specified above
                best_artifact_path = "IrisClassifier"
                joblib.dump(model, "model.pkl")

    print(f"Best model: {best_model_name} with F1-score {best_f1:.4f}")

    # Register best model
    if best_run_id and best_artifact_path:
        # Construct the model_uri correctly
        model_uri = f"runs:/{best_run_id}/{best_artifact_path}"
        mlflow.register_model(model_uri, "IrisClassifier")
        print(f"Registered model {best_model_name} with URI: {model_uri}")


if __name__ == "__main__":
    train_and_log()
