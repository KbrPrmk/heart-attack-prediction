import gradio as gr
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

MODEL_REPO = "KubraParmak/heart-attack-prediction-model"
ARTIFACT_FILE = "heart_attack_artifact.joblib"

artifact_path = hf_hub_download(repo_id=MODEL_REPO, filename=ARTIFACT_FILE)
artifact = joblib.load(artifact_path)

model = artifact["model"]
scaler = artifact["scaler"]
numeric_features = artifact["numeric_features"] 
feature_names = artifact["feature_names"]      


def build_input_row(age, trtbps, chol, thalachh, oldpeak,
                     sex, cp, fbs, restecg, exng, slp, caa, thall):
    row = pd.Series(0.0, index=feature_names, dtype=float)

    numeric_df = pd.DataFrame([{
        "age": age, "trtbps": trtbps, "chol": chol,
        "thalachh": thalachh, "oldpeak": oldpeak,
    }])[numeric_features]
    scaled = scaler.transform(numeric_df)[0]
    for col, val in zip(numeric_features, scaled):
        row[col] = val

    categorical_values = {
        "sex": sex, "cp": cp, "fbs": fbs, "restecg": restecg,
        "exng": exng, "slp": slp, "caa": caa, "thall": thall,
    }
    for col, val in categorical_values.items():
        dummy_col = f"{col}_{val}"
        if dummy_col in row.index:
            row[dummy_col] = 1.0

    return row.values.reshape(1, -1)


def predict(age, trtbps, chol, thalachh, oldpeak,
            sex, cp, fbs, restecg, exng, slp, caa, thall):
    X = build_input_row(age, trtbps, chol, thalachh, oldpeak,
                         sex, cp, fbs, restecg, exng, slp, caa, thall)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    label = "⚠️ Yüksek risk" if pred == 1 else "✅ Düşük risk"
    return f"{label} — tahmini risk olasılığı: %{proba * 100:.1f}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Yaş (age)", value=55),
        gr.Number(label="Dinlenme tansiyonu - trtbps (mmHg)", value=130),
        gr.Number(label="Kolesterol - chol (mg/dl)", value=240),
        gr.Number(label="Maksimum nabız - thalachh", value=150),
        gr.Number(label="Egzersizle ST depresyonu - oldpeak", value=1.0),
        gr.Radio([0, 1], label="Cinsiyet (sex) — 0: Kadın, 1: Erkek", value=1),
        gr.Dropdown([0, 1, 2, 3], label="Göğüs ağrısı tipi (cp)", value=0),
        gr.Radio([0, 1], label="Açlık kan şekeri > 120 mg/dl (fbs)", value=0),
        gr.Dropdown([0, 1, 2], label="Dinlenme EKG sonucu (restecg)", value=0),
        gr.Radio([0, 1], label="Egzersize bağlı anjina (exng)", value=0),
        gr.Dropdown([0, 1, 2], label="ST eğimi (slp)", value=1),
        gr.Dropdown([0, 1, 2, 3, 4], label="Floroskopiyle görülen damar sayısı (caa)", value=0),
        gr.Dropdown([0, 1, 2, 3], label="Talasemi (thall)", value=2),
    ],
    outputs=gr.Textbox(label="Tahmin"),
    title="❤️ Kalp Krizi Riski Tahmini",
    description=(
        "Hasta ölçümlerine göre kalp krizi riskini tahmin eden bir Logistic Regression demosu. "
        "⚠️ Bu sadece eğitim/portfolyo amaçlı bir projedir, tıbbi tavsiye niteliği taşımaz. "
        "Gerçek sağlık kararları için bir hekime danışın."
    ),
)

if __name__ == "__main__":
    demo.launch()