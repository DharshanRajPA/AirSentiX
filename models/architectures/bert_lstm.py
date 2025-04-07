from transformers import TFBertModel
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

def build_bert_lstm(max_len, num_labels):
    input_ids = Input(shape=(max_len,), dtype='int32', name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype='int32', name="attention_mask")

    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

    x = LSTM(128)(bert_output)
    output = Dense(num_labels, activation='softmax')(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    return model
