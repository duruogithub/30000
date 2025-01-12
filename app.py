import os
import joblib
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import tempfile
from flask import Flask, render_template, request, send_file
from dotenv import load_dotenv

# 使用非交互模式，避免启动 Matplotlib GUI
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
THRESHOLD = float(os.getenv("THRESHOLD", 0.14))

# Load model
model_save_path = os.path.join(os.getcwd(), "rf_model.pkl")
model = joblib.load(model_save_path)

# Feature columns and ranges (same as Streamlit version)
feature_columns = [
    "Gender", "Age", "Residence", "BMI", "smoke", "drink", "FX", "BM", "LWY", "FIT"
]

feature_ranges = {
    "Gender": {"type": "categorical", "options": [0, 1]},
    "Age": {"type": "categorical", "options": [0, 1, 2, 3]},
    "Residence": {"type": "categorical", "options": [0, 1]},
    "BMI": {"type": "categorical", "options": [0, 1]},
    "smoke": {"type": "categorical", "options": [0, 1]},
    "drink": {"type": "categorical", "options": [0, 1]},
    "FX": {"type": "categorical", "options": [0, 1], "label": "History of chronic diarrhea"},
    "BM": {"type": "categorical", "options": [0, 1], "label": "History of chronic constipation"},
    "LWY": {"type": "categorical", "options": [0, 1], "label": "History of chronic appendicitis or appendectomy"},
    "FIT": {"type": "categorical", "options": [0, 1]},
}

feature_label_mapping = {
    "FX": "Chronic diarrhea",
    "BM": "Chronic constipation",
    "LWY": "Appendicitis",
}

# Home page route
@app.route("/", methods=["GET", "POST"])
def index():
    shap_plot_path = None

    if request.method == "POST":
        # Collect feature values from the form
        feature_values = []
        for feature in feature_columns:
            value = request.form.get(feature)
            if value is None:
                value = 0  # Use default value if no selection
            feature_values.append(int(value))

        # Convert to DataFrame
        features = pd.DataFrame([feature_values], columns=feature_columns)

        # Predict the class
        if hasattr(model, 'best_estimator_'):
            best_model = model.best_estimator_
        else:
            best_model = model

        predicted_class = best_model.predict(features)[0]
        predicted_proba = best_model.predict_proba(features)[0]
        probability = predicted_proba[1]
        risk_level = "High Risk" if probability > THRESHOLD else "Low Risk"

        # Generate SHAP plot
        try:
            shap_plot_path = generate_shap_plot(best_model, features)
        except Exception as e:
            shap_plot_path = None
            print(f"SHAP plot generation failed: {e}")

        return render_template("index.html", 
                               probability=probability,
                               risk_level=risk_level,
                               shap_plot_path=shap_plot_path,
                               feature_ranges=feature_ranges,
                               feature_columns=feature_columns)

    return render_template("index.html", feature_ranges=feature_ranges, feature_columns=feature_columns)


# Function to generate SHAP plot
def generate_shap_plot(model, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Replace feature labels
    features_with_labels = replace_feature_labels(features)

    # Generate SHAP force plot
    plt.figure(figsize=(15, 6))  # Set figure size
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        features_with_labels,
        matplotlib=True,
    )

    # Define the static directory for saving the plot
    static_dir = os.path.join(app.root_path, 'static', 'plots')
    os.makedirs(static_dir, exist_ok=True)

    shap_plot_path = os.path.join(static_dir, 'shap_plot.png')
    plt.savefig(shap_plot_path)
    plt.clf()  # Clear the figure after saving

    return f"/static/plots/shap_plot.png"


# Function to replace feature labels for SHAP plot
def replace_feature_labels(features):
    renamed_features = features.rename(columns=feature_label_mapping)
    return renamed_features


# Route to serve SHAP plot image
@app.route('/shap_plot')
def serve_shap_plot():
    # Serve the SHAP plot from the generated path
    shap_plot_path = os.path.join(os.getcwd(), 'static', 'plots', 'shap_plot.png')
    return send_file(shap_plot_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=False)  # Set debug=False for production environment