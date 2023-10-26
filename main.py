import pickle

from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib

data=pd.read_csv('Cleaned Data (2).csv')
app = Flask(__name__)

# Load your trained model
ann_model = keras.models.load_model('ANNmodel.h5')
logistic_model = joblib.load('LogisticModel.joblib')
decisionTree_model=joblib.load('DecisionTree.joblib')

with open('random_forestmodel.pkl', 'rb') as file:
    randomForest_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict(probability=None):
    try:
        selected_model = request.form.get('model')
        age = int(request.form.get('Age'))
        t_Stage = int(request.form.get('T Stage'))
        N_Stage = int(request.form.get('N Stage'))
        sixth_Stage = int(request.form.get('6th Stage'))
        differentiate = int(request.form.get('differentiate'))
        Grade = int(request.form.get('Grade'))
        A_Stage = int(request.form.get('A Stage'))
        Tumor_Size = int(request.form.get('Tumor Size'))
        Estrogen_Status = int(request.form.get('Estrogen Status'))
        Progesterone_Status = int(request.form.get('Progesterone Status'))
        Regional_Node_Examined = int(request.form.get('Regional Node Examined'))
        Reginol_Node_Positive = int(request.form.get('Reginol Node Positive'))
        Survival_Months = int(request.form.get('Survival Months'))

        # Create an input feature array
        input_features = np.array([[age, t_Stage, N_Stage, sixth_Stage, differentiate, Grade, A_Stage,
                                    Tumor_Size, Estrogen_Status, Progesterone_Status, Regional_Node_Examined,
                                    Reginol_Node_Positive, Survival_Months]])

        if selected_model == 'ann':
            # Make predictions using ANN model
            input_features_ann = input_features.reshape(1, -1)
            prediction = ann_model.predict(input_features_ann)
            result = {

                'prediction': prediction[0][0]# Adjust this based on your ANN model's output format
            }
        elif selected_model == 'logistic':
            # Make predictions using Logistic Regression model
            prediction = logistic_model.predict(input_features)
            probability = 1 / (1 + np.exp(-prediction))
            result = {
                'prediction': probability[0]  # Adjust this based on your Logistic Regression model's output format
            }
        elif selected_model == 'decision':
            # Make predictions using Logistic Regression model
            prediction = decisionTree_model.predict(input_features)
            result = {
                'prediction': prediction[0]  # Adjust this based on your Logistic Regression model's output format
            }
        elif selected_model == 'random':
            # Make predictions using Logistic Regression model
            prediction = randomForest_model.predict(input_features)
            probability = 1 / (1 + np.exp(-prediction))
            result = {
                'prediction': probability[0]  # Adjust this based on your Logistic Regression model's output format
            }
        else:
            result = {
                'error': 'Invalid model selection'
            }
        return render_template('result.html', result=result)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)

