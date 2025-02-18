from flask import Flask, request, jsonify
import joblib 
import logging


app = Flask(__name__)


model = joblib.load('model.pkl')  


logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  
    features = data['Time','V14','Unamed:0','V25','V12','V19','V17','V10','V1','V26']  
   
    prediction = model.predict([features])
    
  
    logging.info(f'Received request with features: {features}, Prediction: {prediction}')

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)