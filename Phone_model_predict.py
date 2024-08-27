from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('phone_classifier_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['feature1'], data['feature2']]).reshape(1, -1)
    prediction = model.predict(features)
    label = 'Android' if prediction[0] == 1 else 'iPhone'
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
