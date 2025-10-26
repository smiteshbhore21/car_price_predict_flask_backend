from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
car = pd.read_csv("Cleaned_Car.csv")


@app.route("/predict", methods=["GET"])
def predictGet():
    return "<h1> This is POST route </h1>"


@app.route("/getDropDownData", methods=["GET"])
def getDropDownData():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    years = sorted(car["year"].unique(), reverse=True)
    fuel_types = sorted(car["fuel_type"].unique())
    kms_min = int(car["kms_driven"].min())
    kms_max = int(car["kms_driven"].max())
    response = {
        "companies": companies,
        "models": car_models,
        "fuel_types": fuel_types,
        "years": years,
        "kms_min": kms_min,
        "kms_max": kms_max,
    }
    return jsonify(response), 200


@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get("company")
    car_model = request.form.get("car_model")
    year = int(request.form.get("year"))
    fuel_type = request.form.get("fuel_type")
    kms_driven = int(request.form.get("kms_driven"))
    prediction = model.predict(
        pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=["name", "company", "year", "kms_driven", "fuel_type"],
        )
    )
    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
