from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        name = request.form["name"]
        year = int(request.form["year"])
        km_driven = int(request.form["km_driven"])
        fuel = request.form["fuel"]
        seller_type = request.form["seller_type"]
        transmission = request.form["transmission"]
        owner = request.form["owner"]
        engine = int(request.form["engine"])
        power = int(request.form["power"])
        seats = int(request.form["seats"])
        with open("./data/fuel_coder.pkl", "rb") as file:
            fuel_coder = pickle.load(file)
            new_fuel = fuel_coder.transform([fuel])[0]
        with open("./data/seller_type.pkl", "rb") as file:
            seller_type_coder = pickle.load(file)
            new_seller_type = seller_type_coder.transform([seller_type])[0]
        with open("./data/scaled_X.pkl", "rb") as file:
            scaled_X = pickle.load(file)
            t_Value = scaled_X.transform(
                [
                    [
                        year,
                        km_driven,
                        new_fuel,
                        new_seller_type,
                        transmission,
                        owner,
                        engine,
                        power,
                        seats,
                    ]
                ]
            )
        with open("./data/poly.pkl", "rb") as file:
            poly = pickle.load(file)
            final_X = poly.transform(t_Value)
        with open("./data/model.pkl", "rb") as file:
            model = pickle.load(file)
            prediction = model.predict(final_X)
            output = round(prediction[0], 2)
        return render_template(
            "prediction.html",
            name=name,
            prediction_text=output,
        )
    else:
        return render_template("prediction.html")


if __name__ == "__main__":
    app.run()
