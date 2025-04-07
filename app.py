import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
from transformers import BertTokenizer
from utils.ensemble_predictor import ensemble_predict
from utils.evaluation import calculate_urgency, get_emotion

# Setup
app = Flask(__name__)
CORS(app)  # Allow cross-origin for frontend

# Constants
MODEL_PATHS = {
    "lstm": "models/trained/bert_lstm.h5",
    "bilstm": "models/trained/bert_bilstm.h5",
    "cnn": "models/trained/bert_cnn.h5"
}
TOKENIZER_PATH = "preprocessing/bert-base-uncased-tokenizer"
ENCODER_PATH = "preprocessing/label_encoder.pkl"
MAX_LEN = 128

# Load tokenizer, label encoder, models
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

models = {
    name: tf.keras.models.load_model(path, compile=False)
    for name, path in MODEL_PATHS.items()
}

# Prediction API
@app.route("/predict/", methods=["POST"])
def predict():
    data = request.get_json()
    tweet = data.get("text", "")

    if not tweet:
        return jsonify({"error": "No tweet provided."}), 400

    # Tokenization
    encoded = tokenizer(tweet, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="np")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    model_inputs = [input_ids, attention_mask]

    # Ensemble inference
    prediction_scores = ensemble_predict(models, model_inputs)

    # Determine label
    pred_idx = np.argmax(prediction_scores)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(prediction_scores))
    all_scores = {
        label_encoder.inverse_transform([i])[0]: float(score)
        for i, score in enumerate(prediction_scores)
    }

    # Extra insights
    emotional_tone = get_emotion(tweet)
    urgency_score = calculate_urgency(tweet)

    return jsonify({
        "sentiment": pred_label,
        "confidence": confidence,
        "all_scores": all_scores,
        "emotional_tone": emotional_tone,
        "urgency_score": urgency_score
    })

# Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Air-Senti-X backend is running!"})

# Run
if __name__ == "__main__":
    app.run(debug=True, port=8000)
