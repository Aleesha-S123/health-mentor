from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("step_model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    bmi = float(data['BMI'])
    sleep = float(data['SleepHours'])
    activity = float(data['ExerciseLevel'])

    input_data = np.array([[bmi, sleep, activity]])

    prediction = model.predict(input_data)

    return jsonify({
        "suggested_steps": int(prediction[0])
    })

@app.route('/')
def home():
    return "Health ML API is Running!"

if __name__ == "__main__":
    app.run(debug=True)
