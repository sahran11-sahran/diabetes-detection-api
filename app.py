import pickle
import pandas as pd
from Flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = 'diabetes_random_forest_model.pkl'
with open(MODEL_PATH, 'rb') as file:
    artifacts = pickle.load(file)

preprocessor = artifacts['preprocessor']
rf_model = artifacts['rf_model']
threshold = artifacts['best_threshold']
feature_names = artifacts['feature_names']

@app.route("/")
def home():
    return {
        "status": "online",
        "message": "Diabetes Detection API is running"
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        patients_list = data.get('patients', [])
        if not patients_list:
            return jsonify({'error': 'No patient data provided'}), 400
        input_df = pd.DataFrame(patients_list)
        preprocessed_array = preprocessor.transform(input_df)
        preprocessed_df = pd.DataFrame(preprocessed_array, columns=feature_names)
        probabilities = rf_model.predict_proba(preprocessed_df)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            results.append({
                "patient_index": i,
                "diabetes_probability": float(prob),
                "diabetes_prediction": int(pred),
                "prediction_label": 'Positive (Risk Detected)' if pred == 1 else 'Negative'
            })
        return jsonify("status", "success", "predictions", results)
    except Exception as e:
        return jsonify("status", "error", "message", str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
