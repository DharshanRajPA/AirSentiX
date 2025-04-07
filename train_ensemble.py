import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
import pickle

from preprocessing.data_cleaning import preprocess_dataset
from preprocessing.data_split_encode import split_and_encode
from preprocessing.tokenize_bert import bert_tokenize

from models.architectures.bert_lstm import build_bert_lstm
from models.architectures.bert_bilstm import build_bert_bilstm
from models.architectures.bert_cnn import build_bert_cnn

# --- Configuration ---
DATA_PATH = 'dataset/Tweets.csv'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3

# --- Step 1: Preprocess Dataset ---
print("[INFO] Cleaning dataset...")
df = preprocess_dataset(DATA_PATH)

# --- Step 2: Split and Encode Labels ---
print("[INFO] Splitting and encoding labels...")
train_df, test_df, train_labels, test_labels = split_and_encode(df)

# --- Step 3: Tokenize Text using BERT Tokenizer ---
print("[INFO] Tokenizing text using BERT...")
X_train_input_ids, X_train_attention_masks = bert_tokenize(train_df['text'].tolist(), max_len=MAX_LEN)
X_test_input_ids, X_test_attention_masks = bert_tokenize(test_df['text'].tolist(), max_len=MAX_LEN)

# --- Step 4: Build All Models ---
num_labels = len(set(train_labels))

print("[INFO] Building models...")
model_lstm = build_bert_lstm(MAX_LEN, num_labels)
model_bilstm = build_bert_bilstm(MAX_LEN, num_labels)
model_cnn = build_bert_cnn(MAX_LEN, num_labels)

models = [model_lstm, model_bilstm, model_cnn]
model_names = ['bert_lstm', 'bert_bilstm', 'bert_cnn']

# --- Step 5: Compile Models ---
for model in models:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# --- Step 6: Train Each Model ---
for i, model in enumerate(models):
    print(f"[INFO] Training {model_names[i]} model...")
    model.fit(
        [X_train_input_ids, X_train_attention_masks],
        train_labels,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

# --- Step 7: Predict and Average ---
print("[INFO] Predicting with ensemble...")

all_preds = []
for model in models:
    preds = model.predict([X_test_input_ids, X_test_attention_masks])
    all_preds.append(preds)

# Average logits
avg_preds = np.mean(np.array(all_preds), axis=0)

# Get final predicted labels
final_preds = tf.argmax(avg_preds, axis=1).numpy()

# --- Step 8: Classification Report ---
print("[RESULT] Ensemble Classification Report:")
print(classification_report(test_labels, final_preds))

# --- Step 9: Save All Models ---
for i, model in enumerate(models):
    model_path = f'models/saved/{model_names[i]}_model'
    model.save(model_path)
    print(f"[INFO] {model_names[i]} model saved at {model_path}")
