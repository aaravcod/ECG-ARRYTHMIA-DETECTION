import numpy as np
import pandas as pd
from keras.models import load_model
import joblib


model = load_model('./rnn_arrythmia_model.h5')
scaler = joblib.load('./scaler.pkl')
label_encoder = joblib.load('./label_encoder.pkl')

def preprocess_input(data):
    scaled_data = scaler.transform(data)
    scaled_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))
    return scaled_data

def make_prediction(input_data):
    processed_data = preprocess_input(input_data)
    predictions = model.predict(processed_data)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    return predicted_labels, predictions

df = pd.read_csv('./MIT-BIH Arrhythmia Database.csv') 
input_features = df.iloc[:, 2:].values 

predicted_classes, class_probabilities = make_prediction(input_features)

df['Predicted_Type'] = predicted_classes
for i in range(class_probabilities.shape[1]):
    df[f'Probability_Class_{i}'] = class_probabilities[:, i]

df.to_csv('./Prediction_Results.csv', index=False)

print("Predictions completed. Results saved to 'Prediction_Results.csv'.")