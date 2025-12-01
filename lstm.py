import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import os

# Hyperparameters
EMBEDDING_DIM = 300
latent_dim = 300
num_words = 10004
units = 128
max_length = 30  # you can tune this

# Step 1: Load dataset
dataset = load_dataset("findnitai/english-to-hinglish", split="train")
eng_texts = [sample["translation"]["en"] for sample in dataset]
hin_texts = ["<sos> " + sample["translation"]["hi_ng"] + " <eos>" for sample in dataset]

# Step 2: Tokenize
eng_tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
eng_tokenizer.fit_on_texts(eng_texts)
eng_sequences = eng_tokenizer.texts_to_sequences(eng_texts)
encoder_input_data = pad_sequences(eng_sequences, maxlen=max_length, padding='post')

hin_tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
hin_tokenizer.fit_on_texts(hin_texts)
hin_sequences = hin_tokenizer.texts_to_sequences(hin_texts)

decoder_input_data = pad_sequences([seq[:-1] for seq in hin_sequences], maxlen=max_length, padding='post')
decoder_target_data = pad_sequences([seq[1:] for seq in hin_sequences], maxlen=max_length, padding='post')

# Step 3: Load GloVe embeddings
glove_input_file = "/kaggle/input/glove6b300dtxt/glove.6B.300d.txt"
word2vec_output_file = "glove.6B.300d.word2vec.txt"
if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)

glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
for word, i in eng_tokenizer.word_index.items():
    if i > num_words:
        continue
    if word in glove_model:
        embedding_matrix[i] = glove_model[word]

# Step 4: Build Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        encoder_outputs, decoder_outputs = inputs
        score = tf.matmul(decoder_outputs, encoder_outputs, transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, encoder_outputs)
        return context_vector, attention_weights

# Step 5: Model architecture
embedding_layer = Embedding(num_words + 1, latent_dim, weights=[embedding_matrix], trainable=False)

# Encoder
encoder_inputs = Input(shape=(max_length,), name="encoder_input")
encoder_emb = embedding_layer(encoder_inputs)
encoder_lstm_1 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output1, _, _ = encoder_lstm_1(encoder_emb)
encoder_lstm_2 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output2, state_h, state_c = encoder_lstm_2(encoder_output1)

# Decoder
decoder_inputs = Input(shape=(max_length,), name="decoder_input")
decoder_emb_layer = Embedding(num_words, latent_dim, trainable=True)
decoder_emb = decoder_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=[state_h, state_c])

# Attention
attention = AttentionLayer()
context_vector, _ = attention([encoder_output2, decoder_outputs])
decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, context_vector])

# Final dense
decoder_dense = TimeDistributed(Dense(num_words, activation="softmax"))
decoder_final_outputs = decoder_dense(decoder_concat_input)

# Full model
model = Model([encoder_inputs, decoder_inputs], decoder_final_outputs)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Step 6: Train
model.fit(
    [encoder_input_data, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=64,
    epochs=5,
    validation_split=0.1
)

# Step 7: Save model
model.save("english_to_hinglish_attention_glove.h5")
