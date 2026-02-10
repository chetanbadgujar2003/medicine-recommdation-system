from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "svc.pkl")

# ---------- LOAD DATA ----------
description = pd.read_csv(os.path.join(DATASET_DIR, "description.csv"))
precautions = pd.read_csv(os.path.join(DATASET_DIR, "precautions_df.csv"))
medications = pd.read_csv(os.path.join(DATASET_DIR, "medications.csv"))
workout = pd.read_csv(os.path.join(DATASET_DIR, "workout_df.csv"))
diets = pd.read_csv(os.path.join(DATASET_DIR, "diets.csv"))
symptoms_df = pd.read_csv(os.path.join(DATASET_DIR, "symtoms_df.csv"))

# ---------- LOAD MODEL ----------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- HELPER FUNCTIONS ----------
def get_description(disease):
    row = description[description["Disease"] == disease]
    return row["Description"].values[0] if not row.empty else "No description available."

def get_precautions(disease):
    row = precautions[precautions["Disease"] == disease]
    return row.iloc[0, 1:].dropna().tolist() if not row.empty else []

def get_medications(disease):
    row = medications[medications["Disease"] == disease]
    return row["Medication"].tolist() if not row.empty else []

def get_workout(disease):
    row = workout[workout["Disease"] == disease]
    return row["Workout"].tolist() if not row.empty else []

def get_diet(disease):
    row = diets[diets["Disease"] == disease]
    return row["Diet"].tolist() if not row.empty else []

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/developer")
def developer():
    return render_template("developer.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form.get("symptoms")

    if not symptoms:
        return render_template("index.html", message="Please enter symptoms")

    symptoms_list = symptoms.split(",")
    input_vector = np.zeros(len(symptoms_df.columns))

    for symptom in symptoms_list:
        symptom = symptom.strip()
        if symptom in symptoms_df.columns:
            input_vector[symptoms_df.columns.get_loc(symptom)] = 1

    prediction = model.predict([input_vector])[0]

    return render_template(
        "index.html",
        predicted_disease=prediction,
        dis_des=get_description(prediction),
        my_precautions=get_precautions(prediction),
        medications=get_medications(prediction),
        workout=get_workout(prediction),
        my_diet=get_diet(prediction),
    )

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
