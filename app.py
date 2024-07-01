from flask import Flask, render_template, request, jsonify, send_file
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


app = Flask(__name__)

# Load the models
# with open("model.pkl", "rb") as file:
#     linear_model = pickle.load(file)

# with open("model_ridge.pkl", "rb") as file:
#     ridge_model = pickle.load(file)

# with open("model_lasso.pkl", "rb") as file:
#     lasso_model = pickle.load(file)

# with open("model_EN.pkl", "rb") as file:
#     elastic_net_model = pickle.load(file)


with open("model.pkl", "rb") as file:
    Linearmodel = pickle.load(file)


def make_predictions(X):
    linearModelPredictions = Linearmodel.predict(X)
    return linearModelPredictions


def prepare_input(data):
    X = np.array([value for value in data.values()]).reshape(1, -1)
    return X


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predictions", methods=["POST"])
def predict():
    new_data = {}
    if request.method == "POST":
        age = int(request.form["age"])
        weight = int(request.form["weight"])
        crcl = int(request.form["crcl"])
        rate = int(request.form["rate"])
        sex = 1 if request.form["sex"] == 'M' else 0

        new_data.update(dict({'RATE': [rate], 'WEIGHT': [weight], 'AGE': [
            age], 'SEX': [sex], 'CRCL': [crcl]}))
        print(new_data)
        isConvert = prepare_input(new_data)
        notResult = make_predictions(isConvert)
        listResult = notResult.tolist()
        print(listResult)

        return jsonify(listResult)


if __name__ == '__main__':
    app.run(debug=True)
