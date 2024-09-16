# -*- coding: utf-8 -*-
"""task3.py"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure the necessary libraries are installed
# You need to run this in your terminal or command prompt
# pip install kaggle pandas scikit-learn

# Load the dataset
df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')
print(df.head())
print(df.columns)

# Preprocess the Data
label_encoders = {}
for column in ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Verify the DataFrame after encoding
print(df.head())
print(df.columns)

# Split the data into features and target
X = df.drop('Outcome Variable', axis=1)
y = df['Outcome Variable']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Classification Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Making Predictions with new data
def predict_disease(new_data):
    new_data_encoded = {}
    for column, value in new_data.items():
        if column in label_encoders:
            new_data_encoded[column] = label_encoders[column].transform([value])[0]
        else:
            new_data_encoded[column] = value

    new_data_df = pd.DataFrame([new_data_encoded], columns=X.columns)
    prediction = model.predict(new_data_df)
    prediction_decoded = label_encoders['Outcome Variable'].inverse_transform(prediction)
    return prediction_decoded[0]

# Example predictions
new_data_1 = {
    'Disease': 'Influenza',
    'Fever': 'Yes',
    'Cough': 'No',
    'Fatigue': 'Yes',
    'Difficulty Breathing': 'Yes',
    'Age': 20,
    'Gender': 'Female',
    'Blood Pressure': 'Low',
    'Cholesterol Level': 'Normal'
}

new_data_2 = {
    'Disease': 'Common Cold',
    'Fever': 'No',
    'Cough': 'Yes',
    'Fatigue': 'Yes',
    'Difficulty Breathing': 'No',
    'Age': 25,
    'Gender': 'Female',
    'Blood Pressure': 'Normal',
    'Cholesterol Level': 'Normal'
}

print(f'Prediction for new data 1: {predict_disease(new_data_1)}')
print(f'Prediction for new data 2: {predict_disease(new_data_2)}')
