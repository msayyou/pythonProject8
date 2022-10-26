import pickle
import pandas as pd
from flask import Flask, request, render_template

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@flask_app.route('/')
def home():
    return "application fonctionne bien"


@flask_app.route("/prevision", methods=["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prevision = model.predict(query_df)
    return f"The prediction for this individual is {round(prevision[0], 2)}!"


if __name__ == "__main__":
    flask_app.run(debug=True)
