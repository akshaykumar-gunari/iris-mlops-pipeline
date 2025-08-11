from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import joblib

app = Flask(__name__)

# Load the registered best model from MLflow Model Registry
model_name = "IrisClassifier"  # Replace if you used a different name
model_version = 1  # Use the appropriate version

model_uri = f"models:/{model_name}/{model_version}"
# model = mlflow.sklearn.load_model(model_uri)
model = joblib.load("model.pkl")

# Define feature columns expected in input JSON
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # data = request.json
        data = {
        "sepal length (cm)": 100,
        "sepal width (cm)": 60,
        "petal length (cm)": 70,
        "petal width (cm)": 85
        }
        data = request.json

        # Validate input contains all required features
        if not all(feature in data for feature in FEATURE_COLUMNS):
            return jsonify({"error": f"Input JSON must contain {FEATURE_COLUMNS}"}), 400

        # Convert input data to DataFrame for model
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        # Get prediction
        preds = model.predict(input_df)
        preds_proba = model.predict_proba(input_df)

        # Format response
        response = {
            "prediction": int(preds[0]),
            "probabilities": preds_proba[0].tolist()
        }
        print("Result: ", response)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # predict()

