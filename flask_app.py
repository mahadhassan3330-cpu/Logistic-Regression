
from flask import Flask, request, render_template_string
from pathlib import Path
import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DATASETS = {
    "cancer": {
        "file": "breast_cancer_model.joblib",
        "title": "Breast Cancer Classification",
        "description": "Predict benign or malignant using 30 measurements."
    },
    "wine": {
        "file": "wine_model.joblib",
        "title": "Wine Recognition",
        "description": "Classify wine using 13 chemical measurements."
    }
}

for key, config in DATASETS.items():
    saved = joblib.load(MODEL_DIR / config["file"])
    config["model"] = saved["model"]
    config["features"] = saved["feature_names"]
    config["target_names"] = saved["target_names"]


STYLE = """
<style>
* { box-sizing: border-box; }
body {
    margin: 0; background: #f3f6fb; color: #172033;
    font-family: Arial, sans-serif;
}
nav {
    padding: 22px 6%; background: white;
    border-bottom: 1px solid #e2e8f0;
    font-size: 20px; font-weight: bold;
}
nav span { color: #315ce8; }
.hero {
    background: linear-gradient(120deg, #172554, #315ce8);
    color: white; padding: 38px 6%;
}
.hero h1 { margin: 0 0 12px; }
.hero p { color: #dbeafe; line-height: 1.6; }
.container { max-width: 1100px; margin: auto; padding: 28px 20px; }
.card {
    background: white; padding: 26px; border-radius: 15px;
    border: 1px solid #e2e8f0; margin-bottom: 22px;
}
.grid {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px; margin-top: 22px;
}
.model-grid {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
}
label { display: block; font-size: 13px; font-weight: bold; margin-bottom: 8px; }
input {
    width: 100%; min-width: 0; padding: 12px;
    border: 1px solid #cbd5e1; border-radius: 8px; font-size: 14px;
}
input:focus { outline: 2px solid #bfdbfe; border-color: #315ce8; }
.button {
    display: inline-block; padding: 12px 17px; margin-top: 22px;
    border: 0; border-radius: 8px; background: #315ce8;
    color: white; font-size: 14px; font-weight: bold;
    text-decoration: none; cursor: pointer;
}
.secondary { background: #e8edff; color: #2449c7; }
.muted { color: #64748b; line-height: 1.6; }
.result {
    margin-top: 25px; padding: 22px; background: #eff6ff;
    border: 1px solid #bfdbfe; border-radius: 10px;
}
.result h2 { color: #1d4ed8; overflow-wrap: anywhere; }
.error {
    padding: 14px; margin-top: 20px;
    background: #fef2f2; color: #991b1b; border-radius: 8px;
}
footer { text-align: center; color: #64748b; padding: 24px; }
@media (max-width: 760px) {
    .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .model-grid { grid-template-columns: 1fr; }
}
@media (max-width: 460px) {
    .grid { grid-template-columns: 1fr; }
    .card { padding: 18px; }
}
</style>
"""

HOME = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Logistic Regression Lab</title>
{{ style|safe }}
</head>
<body>
<nav>Logistic<span>Regression Lab</span></nav>
<section class="hero">
<h1>Classification Prediction Studio</h1>
<p>Explore two trained Logistic Regression models through an
interactive machine learning dashboard.</p>
</section>
<main class="container">
<div class="card">
<h2>Choose a dataset</h2>
<p class="muted">Select a model to open its prediction form.</p>
<div class="model-grid">
<div class="card">
<h3>Breast Cancer</h3>
<p class="muted">Classify a tumor as benign or malignant.</p>
<a class="button" href="/cancer">Open Model →</a>
</div>
<div class="card">
<h3>Wine Recognition</h3>
<p class="muted">Predict one of three wine classes.</p>
<a class="button" href="/wine">Open Model →</a>
</div>
</div>
</div>
<div class="card">
<h3>About this project</h3>
<p class="muted">Both models use Scikit-learn Logistic Regression
with StandardScaler. Evaluation includes accuracy, precision,
recall and F1-score.</p>
</div>
</main>
<footer>ANN Assignment · Logistic Regression · Flask</footer>
</body>
</html>
"""

PREDICTION = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ config.title }}</title>
{{ style|safe }}
</head>
<body>
<nav><a href="/" style="text-decoration:none;color:inherit">
Logistic<span>Regression Lab</span></a></nav>
<section class="hero">
<h1>{{ config.title }}</h1>
<p>{{ config.description }}</p>
</section>
<main class="container">
<div class="card">
<h2>Enter feature values</h2>
<p class="muted">Use realistic values in the original dataset's
measurement units.</p>
<form method="POST">
<div class="grid">
{% for feature in config.features %}
<div>
<label for="{{ feature }}">{{ feature.replace("_", " ").title() }}</label>
<input type="number" step="any" required
id="{{ feature }}" name="{{ feature }}"
value="{{ request.form.get(feature, '') }}">
</div>
{% endfor %}
</div>
<button class="button" type="submit">Predict Class →</button>
<a class="button secondary" href="{{ request.path }}">Reset</a>
</form>

{% if result %}
<div class="result">
<h2>Prediction: {{ result }}</h2>
<h3>Class probabilities</h3>
{% for name, probability in probabilities %}
<p>{{ name }}: {{ "%.2f"|format(probability * 100) }}%</p>
{% endfor %}
<p class="muted">Probabilities are model estimates, not guarantees.</p>
</div>
{% endif %}
{% if error %}
<div class="error">{{ error }}</div>
{% endif %}
</div>
</main>
<footer>Scikit-learn · Logistic Regression</footer>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HOME, style=STYLE)


@app.route("/<model_name>", methods=["GET", "POST"])
def predict(model_name):
    if model_name not in DATASETS:
        return "Model not found", 404

    config = DATASETS[model_name]
    result = None
    probabilities = []
    error = None

    if request.method == "POST":
        try:
            values = [
                float(request.form[feature])
                for feature in config["features"]
            ]

            if not np.isfinite(values).all():
                raise ValueError("Inputs must be finite numbers.")

            inputs = np.array(values, dtype=float).reshape(1, -1)
            prediction = int(config["model"].predict(inputs)[0])
            scores = config["model"].predict_proba(inputs)[0]

            result = config["target_names"][prediction]
            probabilities = list(zip(config["target_names"], scores))

        except (ValueError, KeyError):
            error = "Please enter valid numbers in every field."
        except Exception:
            error = "Prediction failed. Please check the input values."

    return render_template_string(
        PREDICTION,
        style=STYLE,
        config=config,
        request=request,
        result=result,
        probabilities=probabilities,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
