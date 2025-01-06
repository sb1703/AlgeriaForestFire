from logging import debug
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/rd.pkl', 'rb'))
standard_scaler_model = pickle.load(open('models/sc.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = request.form['Classes']
        Region = request.form['Region']

        new_data_scaled = standard_scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
