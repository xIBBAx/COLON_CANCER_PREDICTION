from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from datetime import datetime

from model import train_linear

app = Flask(__name__)

trained_model_name = "model.pkl"

with open(trained_model_name, "rb") as file:
    Linearmodel = pickle.load(file)


def make_predictions_linear(X):
    linearModelPredictions = Linearmodel.predict(X)
    return linearModelPredictions


with open("model_lasso.pkl", "rb") as file:
    Lassomodel = pickle.load(file)


def make_predictions_lasso(X):
    LassomodelPredictions = Lassomodel.predict(X)
    return LassomodelPredictions


with open("model_ridge.pkl", "rb") as file:
    ridgemodel = pickle.load(file)


def make_predictions_ridge(X):
    ridgemodelPredictions = ridgemodel.predict(X)
    return ridgemodelPredictions


with open("model_EN.pkl", "rb") as file:
    ENmodel = pickle.load(file)


def make_predictions_EN(X):
    ENmodelPredictions = ENmodel.predict(X)
    return ENmodelPredictions


def prepare_input(data):
    X = np.array([value for value in data.values()]).reshape(1, -1)
    return X


def replace_null_and_zero_with_mean(data, columns):
    for col in columns:
        if data[col].isnull().any() or (data[col] == 0).any():
            mean_value = data[col].replace(0, pd.NA).mean()
            data[col].fillna(mean_value, inplace=True)

# Function to repeat data until 5000 patients, excluding a column


def repeat_data_until_5000(data, exclude_column):
    data_excluded = data.drop(columns=[exclude_column])
    total_patients = len(data_excluded)
    repetitions_needed = (5000 // total_patients) + 1
    repeated_data = pd.concat(
        [data_excluded] * repetitions_needed, ignore_index=True)
    return repeated_data[:5000]


# def generate_plot(prediction_results, filename):
#     # Generate your plot here
#     output_features = ['DV 1st day', '1', '2', '4', '8', '12', 'DV 2nd day', 'DV 3rd day', 'DV 4th day', 'DV 7th day', '145', '146', '148', '150', '152', '156', '167']
#     plt.plot(output_features, prediction_results, label='Predicted', marker='o', linestyle='-')
#     plt.legend()
#     plt.title('Comparison of Actual and Predicted output')
#     plt.xlabel('timing')
#     plt.xticks(rotation=90)
#     plt.ylabel('DV')
#     # Save the plot to a file
#     plt.savefig(filename, format='png')
#     plt.close()


@app.route('/')
def home():
    return render_template("index.html")

# @app.route('/plot', methods=["POST"])
# def plot():
#     if request.method == "POST":
#         # Get prediction results from the request
#         prediction_results = request.json['predictions']
#         # Generate a unique filename for the plot
#         plot_filename = 'plot.png'
#         # Specify the directory to save the plot image
#         plot_directory = 'static'
#         # Create the directory if it doesn't exist
#         os.makedirs(plot_directory, exist_ok=True)
#         # Generate and save the plot
#         plot_path = os.path.join(plot_directory, plot_filename)
#         generate_plot(prediction_results, plot_path)
#         # Return the plot file
#         return send_file(plot_path, mimetype='image/png')


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
        notResult_linear = make_predictions_linear(isConvert)
        notResult_lasso = make_predictions_lasso(isConvert)
        notResult_EN = make_predictions_EN(isConvert)
        notResult_ridge = make_predictions_ridge(isConvert)

        listResult_linear = notResult_linear[0].tolist()
        listResult_EN = notResult_EN[0].tolist()
        listResult_lasso = notResult_lasso[0].tolist()
        listResult_ridge = notResult_ridge[0].tolist()

        print([listResult_linear, listResult_lasso,
              listResult_ridge, listResult_EN])

        return jsonify([listResult_linear, listResult_lasso, listResult_ridge, listResult_EN])


@app.route('/train', methods=['POST'])
def preprocess_and_train():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    model_type = request.form.get('model_type', '')

    if file and file.filename.endswith('.csv'):
        # Read data from CSV file
        data = pd.read_csv(file)

        # Preprocess data
        data.drop(columns=['DV 5th day', 'DV 6th day',
                  'DV 8th day'], inplace=True)
        data['SEX'] = data['SEX'].replace({'M': 0, 'F': 1})
        data['GROUP'] = data['Group'].str.replace(' mg', '').astype(int)
        # data = replace_null_and_zero_with_mean(data, columns=data.columns)
        processed_data = repeat_data_until_5000(data, exclude_column='ID')
        label_encoder = LabelEncoder()
        data['GROUP'] = label_encoder.fit_transform(data['Group'])
        data.drop(["Group"], axis=1, inplace=True)

        processed_data.to_csv("processed_data.csv", index=False)
        # Train the model
        message = train_model(processed_data, model_type)

        return message
    else:
        return "Invalid file format"


def train_model(data, model_type):

    output_features = ['DV 1st day', '1', '2', '4', '8', '12', 'DV 2nd day', 'DV 3rd day',
                       'DV 4th day', 'DV 7th day', '145', '146', '148', '150', '152', '156', '167']
    input_features = ['RATE', 'WEIGHT', 'AGE', 'SEX', 'CRCL']

    X = data[input_features]
    y = data[output_features]

    X_train, X_test, y1_train, y1_test = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=1)

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge()
    elif model_type == 'lasso':
        model = Lasso()
    elif model_type == 'elastic_net':
        model = ElasticNet()
    else:
        return "Invalid model type"

    model.fit(X_train, y1_train)

    # Save the trained model
    with open(f'model_{model_type}.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Construct the message
    model_name = model_type.capitalize() + " Regression"
    csv_link = "<a href='processed_data.csv'>processed_data.csv</a>"
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"The {model_name} has been trained with CSV {csv_link} on {date_time}"
    print(message)

    return f"<p> {message} </p>"


def replace_null_and_zero_with_mean(data, columns):
    imputer = SimpleImputer(strategy='mean')
    data[columns] = imputer.fit_transform(data[columns])
    return data


def repeat_data_until_5000(data, exclude_column):
    data_excluded = data.drop(columns=[exclude_column])
    total_patients = len(data_excluded)
    repetitions_needed = (5000 // total_patients) + 1
    repeated_data = pd.concat(
        [data_excluded] * repetitions_needed, ignore_index=True)
    return repeated_data[:5000]


if __name__ == '__main__':
    app.run(debug=True)
