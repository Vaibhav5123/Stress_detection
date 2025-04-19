from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('Stress_Detection.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data['body_temperature'] = (data['body_temperature'] * 9/5) + 32
    data['blood_oxygen'] = data['blood_oxygen']
    data['heart_rate'] = data['heart_rate'] / 2
    features = [[
        data['body_temperature'],
        data['blood_oxygen'],
        data['heart_rate']
    ]]
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
