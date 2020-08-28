from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
)
import numpy as np


def get_scaled_data(method="None", p_degree=None, input_data=None):
    if method == "Standard":
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == "MinMax":
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == "Log":
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data

    if p_degree != None:
        scaled_data = PolynomialFeatures(
            degree=p_degree, include_bias=False
        ).fit_transform(scaled_data)

    return scaled_data


from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():

    return render_template("index.html")


import pandas as pd
import joblib


@app.route("/result")
def result():
    ridge = joblib.load("./ridge_model.pkl")

    temp = float(request.args.get("기온"))
    humi = int(request.args.get("습도"))
    cloud = int(request.args.get("구름"))
    sun = int(request.args.get("일조"))
    month = int(request.args.get("월"))
    disco = float(0.81 * temp + 0.01 * humi * (0.99 * temp - 14.3) + 46.3)

    feature = pd.DataFrame(
        data={
            "평균기온": [temp],
            "평균상대습도": [humi],
            "cloud_amount": [cloud],
            "sun_amount": [sun],
            "월_1": [0],
            "월_2": [0],
            "월_3": [0],
            "월_4": [0],
            "월_5": [0],
            "월_6": [0],
            "월_7": [0],
            "월_8": [0],
            "월_9": [0],
            "월_10": [0],
            "월_11": [0],
            "월_12": [0],
            "불쾌지수": [disco],
        }
    )
    feature = feature.astype(
        {
            "평균기온": "float64",
            "월_1": "int64",
            "월_2": "int64",
            "월_3": "int64",
            "월_4": "int64",
            "월_5": "int64",
            "월_6": "int64",
            "월_7": "int64",
            "월_8": "int64",
            "월_9": "int64",
            "월_10": "int64",
            "월_11": "int64",
            "월_12": "int64",
        }
    )

    feature[f"월_{month}"] = 1
    file_name = "./StandardScaler.pkl"
    sc = joblib.load(file_name)
    scaled_feature = sc.transform(feature)
    pred = ridge.predict(scaled_feature)[0]

    if sun == 1:
        sun = "적다"
    elif sun == 2:
        sun = "보통"
    else:
        sun = "많다"

    if cloud == 1:
        cloud = "적다"
    elif cloud == 2:
        cloud = "보통"
    else:
        cloud = "많다"
    result = {
        "temp": temp,
        "humi": humi,
        "cloud": cloud,
        "sun": sun,
        "month": month,
        "disco": int(disco),
        "pred": int(np.round(np.expm1(pred), 0)),
    }
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=8089)

