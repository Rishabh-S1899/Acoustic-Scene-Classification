# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:59:10 2024

@author: chetan kukreja
"""

import os
import glob
import pickle
import numpy as np
import soundfile
import openl3
import tensorflow as tf
import logging
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # Corrected import
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def get_file_paths_in_batches(folder_path, batch_size=32):
    all_file_paths = []
    for _, _, files in os.walk(folder_path):
        for file in files:
            all_file_paths.append(file)
    logging.info(f"Total files found: {len(all_file_paths)}")
    total_files = len(all_file_paths)
    start_index = 0

    while start_index < total_files:
        end_index = start_index + batch_size
        batch = all_file_paths[start_index:end_index]
        start_index = end_index
        
        yield batch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def find_indexes(subset):
    superset = ['stft_magnitude', 'batch_normalization', 'conv2d', 'batch_normalization_1', 'activation', 'conv2d_1', 'batch_normalization_2', 'activation_1', 'max_pooling2d', 'conv2d_2', 'batch_normalization_3', 'activation_2', 'conv2d_3', 'batch_normalization_4', 'activation_3', 'max_pooling2d_1', 'conv2d_4', 'batch_normalization_5', 'activation_4', 'conv2d_5', 'batch_normalization_6', 'activation_5', 'max_pooling2d_2', 'conv2d_6', 'batch_normalization_7', 'activation_6', 'audio_embedding_layer', 'max_pooling2d_3', 'flatten']
    indexes = []
    for element in subset:
        try:
            index = superset.index(element)
            indexes.append(index)
            
        except ValueError:
            logging.warning(f"Element '{element}' not found in the superset.")
    del superset
    return indexes

def model_loader(input_repr="mel256", content_type="env", embedding_size=512, hop_size=1, files=[], embeddings_of=['activation_1', 'activation_3', 'activation_5', 'flatten']):
    logging.info("Loading audio embedding model...")
    model = openl3.models.load_audio_embedding_model(input_repr, content_type, embedding_size)
    logging.info('Building model...')
    outs = []
    y = model.input
    layers = model.layers[1:]
    for layer in layers:
        y = layer(y)
        outs.append(y)
    models=type(model)(model.inputs, outs)
    del model,outs
    return models 
def datamaker(stepmodel ,operation='sum', folder_path='./data', files=[] , embeddings_of=['activation_1', 'activation_3', 'activation_5', 'flatten'], hop_size=1):
    for f in files:
        Y.append(list(df[df['filename'] ==f]['category'])[0])
        f = os.path.join(folder_path, f)
        y, sr = soundfile.read(f)
        logging.info(f'Processing {f} - Sample Rate: {sr}, Shape: {y.shape}')
        batches = openl3.core._preprocess_audio_batch(y, sr, hop_size=hop_size)
        Z = stepmodel.predict(batches)
        del batches
        for i in find_indexes(embeddings_of):
            if i!=28:
                if operation == 'sum':
                    pooled_tensor = tf.keras.layers.AveragePooling2D(pool_size=(Z[i].shape[1], Z[i].shape[2]), strides=(1, 1), padding='valid')(tf.convert_to_tensor(Z[i]))
                    print(pooled_tensor.shape)
                    X[str(i)].append(pooled_tensor.numpy().reshape(-1))  # Convert to NumPy array if necessary, and reshape if needed               
                elif operation == 'max':
                    pooled_tensor = tf.keras.layers.MaxPooling2D(pool_size=(Z[i].shape[1], Z[i].shape[2]), strides=(1, 1), padding='valid')(tf.convert_to_tensor(Z[i]))
                    print(f"Max Pooled Tensor Shape for {i}: {pooled_tensor.shape}")
                    X[str(i)].append(pooled_tensor.numpy().reshape(-1))
            else:
                X[str(i)].append(Z[i].reshape(-1))
        del Z
    del files
def model_maker(input_dim,num_classes=50):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dense(num_classes, activation='softmax')
    ])
    # Define the optimizer and learning rate scheduler
    initial_lr = 0.00001
    """lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )"""
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def OnehotEncoding(Y):
    # One-hot encode the labels
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    num_classes = len(np.unique(Y))
    ohe = OneHotEncoder(sparse_output=False)
    Y = ohe.fit_transform(Y.reshape(-1, 1))
    return Y
X = {}
Y=[]
label_path = "./esc50/esc50/meta/esc50.csv"
df=pd.read_csv(label_path)
folderpath = "./esc50/esc50/audio_32k"
epochs = 100
embeddings_of = ['activation_1', 'activation_3', 'activation_5','flatten']
#embeddings_of = ['stft_magnitude', 'batch_normalization', 'conv2d', 'batch_normalization_1', 'activation', 'conv2d_1', 'batch_normalization_2', 'activation_1', 'max_pooling2d', 'conv2d_2', 'batch_normalization_3', 'activation_2', 'conv2d_3', 'batch_normalization_4', 'activation_3', 'max_pooling2d_1', 'conv2d_4', 'batch_normalization_5', 'activation_4', 'conv2d_5', 'batch_normalization_6', 'activation_5', 'max_pooling2d_2', 'conv2d_6', 'batch_normalization_7', 'activation_6', 'audio_embedding_layer', 'max_pooling2d_3', 'flatten']
stepmodel = model_loader(embeddings_of=embeddings_of)
operation='sum'
batch_num = 0
epoch_training_loss = {'7': 0, '14': 0, '21': 0, '28':0}
models={}
histories = {}

for i in find_indexes(embeddings_of):
    X.update({f'{i}':[]})
for file_paths_batch in get_file_paths_in_batches(folderpath, batch_size=32):
    datamaker(stepmodel,operation,folder_path=folderpath,files=file_paths_batch, embeddings_of=embeddings_of) 
Y=OnehotEncoding(Y)
for i in find_indexes(embeddings_of):
    models.update({f'{i}':None})
for i in find_indexes(embeddings_of):
    X_train, X_test, y_train, y_test = train_test_split(np.vstack(X[str(i)]), Y, test_size=0.2, random_state=42)
    models[str(i)]=model_maker(X_train.shape[1],num_classes=Y.shape[1])
    histories[str(i)] = models[str(i)].fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
def plot_losses(histories):
    for model_id, history in histories.items():
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss for Model {model_id}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Accuracy for Model {model_id}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
plot_losses(histories)
os.chdir("./codes")
for i in models.keys():
    models[str(i)].save(f"classifier_maam_SUM_MAX_{i}.h5")