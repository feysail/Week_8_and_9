from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime
import logging


app = Flask(__name__)


model = joblib.load('model.pkl')  

logging.basicConfig(level=logging.INFO)


fraud_data = pd.read_csv('Fraud_Data.csv')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    
    prediction = model.predict([features])
    logging.info(f'Received request with features: {features}, Prediction: {prediction}')

    return jsonify({'prediction': int(prediction[0])})

@app.route('/summary', methods=['GET'])
def summary():
    total_transactions = len(fraud_data)
    total_fraud_cases = fraud_data['class'].sum()
    fraud_percentage = (total_fraud_cases / total_transactions) * 100

    summary_stats = {
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage
    }

    return jsonify(summary_stats)

@app.route('/fraud_trends', methods=['GET'])
def fraud_trends():
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    trends = fraud_data.groupby(fraud_data['purchase_time'].dt.date)['class'].sum().reset_index()
    trends.columns = ['date', 'fraud_cases']
    
    return trends.to_json(orient='records')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)