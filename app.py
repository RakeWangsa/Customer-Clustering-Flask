import pandas as pd
from pyexpat import features
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction==0:
        hasil="Pelanggan Silver"
    elif prediction==1:
        hasil="Pelanggan Platinum"
    elif prediction==2:
        hasil="Pelanggan Gold"    
    return render_template("index.html", prediction_text = "{}".format(hasil))

if __name__ == "__main__":
    app.run(debug=True)