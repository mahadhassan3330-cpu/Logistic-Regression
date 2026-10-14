
import joblib
import numpy as np
import gradio as gr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

cancer_data = joblib.load(MODEL_DIR / "breast_cancer_model.joblib")
wine_data = joblib.load(MODEL_DIR / "wine_model.joblib")


def make_prediction(data, values):
    try:
        if any(value is None for value in values):
            return "Please complete every input field."

        inputs = np.array(values, dtype=float).reshape(1, -1)

        if not np.isfinite(inputs).all():
            return "Please enter valid finite numbers."

        model = data["model"]
        prediction = int(model.predict(inputs)[0])
        probabilities = model.predict_proba(inputs)[0]
        class_names = data["target_names"]
        predicted_name = class_names[prediction]

        probability_text = "\n".join(
            f"- {class_names[i]}: {probability:.2%}"
            for i, probability in enumerate(probabilities)
        )

        return (
            f"### Prediction: {predicted_name}\n\n"
            f"**Class probabilities**\n\n{probability_text}"
        )

    except Exception:
        return "Prediction failed. Please check your inputs."


def create_tab(data, description):
    inputs = []

    gr.Markdown(description)

    features = data["feature_names"]

    for start in range(0, len(features), 3):
        with gr.Row():
            for feature in features[start:start + 3]:
                field = gr.Number(
                    label=feature.replace("_", " ").title(),
                    value=0,
                    precision=5
                )
                inputs.append(field)

    button = gr.Button("Predict Class", variant="primary")
    output = gr.Markdown()

    button.click(
        fn=lambda *values: make_prediction(data, values),
        inputs=inputs,
        outputs=output
    )


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Logistic Regression Lab"
) as demo:

    gr.Markdown(
        """
        # Logistic Regression Lab
        ### ANN Assignment — Classification Models

        Select a dataset, enter its numerical features,
        and predict the corresponding class.
        """
    )

    with gr.Tabs():
        with gr.Tab("Breast Cancer"):
            create_tab(
                cancer_data,
                """
                ## Breast Cancer Classification

                Predict benign or malignant from 30 measurements.
                The inputs are measurements in the dataset's original
                units; enter realistic values for meaningful results.
                """
            )

        with gr.Tab("Wine Recognition"):
            create_tab(
                wine_data,
                """
                ## Wine Classification

                Predict one of three wine classes using 13 chemical
                measurements. Enter values in the dataset's original
                measurement units.
                """
            )

    gr.Markdown(
        "---\n"
        "Algorithm: Logistic Regression · "
        "Scaling: StandardScaler · Library: Scikit-learn"
    )


if __name__ == "__main__":
    demo.launch()
