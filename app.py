from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from keras.models import load_model
import keras.losses

app = Flask(__name__)

# Load trained house price model
model = load_model(
    "model.h5",
    custom_objects={
        "mse": keras.losses.MeanSquaredError(),
        "mean_squared_error": keras.losses.MeanSquaredError(),
        "mae": keras.losses.MeanAbsoluteError(),
    },
    safe_mode=False
)

# Load scaler used during training
scaler = joblib.load("scaler.pkl")
model= load_model("model.h5")
# Feature names
feature_labels = [
    "Overall Quality (1-10)",
    "Above Ground Living Area (sq ft)",
    "Garage Cars",
    "Total Basement Area (sq ft)",
    "Full Bathrooms",
    "Year Built",
    "Lot Area (sq ft)"
]

@app.route('/')
def home():
    return render_template('index.html', feature_labels=feature_labels)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = []
        for label in feature_labels:
            value = float(request.form[label])
            features.append(value)

        features = np.array(features).reshape(1, -1)

        # Scale features
        features = scaler.transform(features)

        # Predict house price
        prediction = model.predict(features)[0][0]

        # Format prediction as currency
        result = f"${prediction:,.2f}"

        return render_template(
            'index.html',
            feature_labels=feature_labels,
            prediction_text=f"Predicted House Price: {result}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            feature_labels=feature_labels,
            prediction_text="Error in prediction. Please check your input values."
        )

if __name__ == "__main__":
    app.run(debug=True)
