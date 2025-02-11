from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Add a Homepage Route
@app.route('/')
def home():
    return "Welcome to the Churn Prediction API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max() * 100  # Get confidence score
    return jsonify({
        'churn_prediction': int(prediction),
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)