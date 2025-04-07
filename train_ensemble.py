import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models.architectures.bert_lstm import build_bert_lstm_model
from models.architectures.bert_bilstm import build_bert_bilstm_model
from models.architectures.bert_cnn import build_bert_cnn_model

# Constants
MODEL_DIR = "models/trained"
TOKENIZER_PATH = "preprocessing/bert-base-uncased-tokenizer"
ENCODER_PATH = "preprocessing/label_encoder.pkl"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load and preprocess data
df = pd.read_csv("data/tweets.csv")  # Assumes CSV has 'text' and 'label' columns
texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

# 2. Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="np")

# 3. Label encoding
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Save tokenizer and encoder
tokenizer.save_pretrained(TOKENIZER_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

# 4. Train-test split
X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    inputs["input_ids"], inputs["attention_mask"], labels_encoded, test_size=0.2, random_state=42
)

# 5. Prepare inputs
train_inputs = [X_train_ids, X_train_mask]
test_inputs = [X_test_ids, X_test_mask]
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
num_classes = y_train.shape[1]

# 6. Build and train models
def train_and_save(model_builder, name):
    print(f"Training {name}...")
    model = model_builder(num_classes)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    checkpoint = ModelCheckpoint(f"{MODEL_DIR}/{name}.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    
    model.fit(train_inputs, y_train,
              validation_data=(test_inputs, y_test),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[checkpoint, early_stop])

    print(f"{name} training complete!\n")

# Train all models
train_and_save(build_bert_lstm_model, "bert_lstm")
train_and_save(build_bert_bilstm_model, "bert_bilstm")
train_and_save(build_bert_cnn_model, "bert_cnn")
