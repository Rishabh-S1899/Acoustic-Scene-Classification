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

import matplotlib.pyplot as plt
import numpy as np


def class_distribution(data_list):
    """
    Computes the per-class distribution and plots a histogram for a given list.
    
    Args:
        data_list (list): A list of data points (strings or integers).
        
    Returns:
        dict: A dictionary containing the per-class distribution.
    """
    # Get the unique classes
    unique_classes = list(set(data_list))
    
    # Create a dictionary to store the class counts
    class_counts = {cls: 0 for cls in unique_classes}
    
    # Count the occurrences of each class
    for item in data_list:
        class_counts[item] += 1
    
    # Convert the counts to a sorted list of tuples
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print the per-class distribution
    print("Per-class distribution:")
    for cls, count in sorted_counts:
        print(f"{cls}: {count}")
    
    # Plot the histogram
    classes = [cls for cls, _ in sorted_counts]
    counts = [count for _, count in sorted_counts]
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution Histogram")
    plt.xticks(rotation=90)
    plt.show()
    
    return class_counts

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
    superset = ['melspectrogram', 'batch_normalization', 'conv2d', 'batch_normalization_1', 'activation', 'conv2d_1', 'batch_normalization_2', 'activation_1', 'max_pooling2d', 'conv2d_2', 'batch_normalization_3', 'activation_2', 'conv2d_3', 'batch_normalization_4', 'activation_3', 'max_pooling2d_1', 'conv2d_4', 'batch_normalization_5', 'activation_4', 'conv2d_5', 'batch_normalization_6', 'activation_5', 'max_pooling2d_2', 'conv2d_6', 'batch_normalization_7', 'activation_6', 'audio_embedding_layer', 'max_pooling2d_3', 'flatten']
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
def datamaker(X,Y,stepmodel ,operation='sum', folder_path='./data', files=[] , embeddings_of=['activation_1', 'activation_3', 'activation_5', 'flatten'], hop_size=1):
    for f in files:
        
        a=f.split("-")
        for i in Y.keys():
            if(i=='location'):
                Y[i].append(a[1])
            if(i=='scene'):
                Y[i].append(a[0])
            if(i=='device'):
                Y[i].append(a[4].split('.')[0])
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
def model_maker(input_dim,num_classes=50,class_weights=None):
    if input_dim < 1000:
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,), name='NN_dense1'),
            Dense(num_classes, activation='softmax', name='NN_dense2')
        ])
    else:
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,), name='NN_dense1'),
            Dense(128, activation='relu', name='NN_dense2'),
            Dense(num_classes, activation='softmax', name='NN_dense3')
        ])
    # Define the optimizer and learning rate scheduler
    initial_lr = 0.00001
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'],)
    return model

def OnehotEncoding(Y,classifier):
    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)
    num_classes = len(np.unique(Y_encoded))

    ohe = OneHotEncoder(sparse_output=False)
    Y_encoded_reshaped = Y_encoded.reshape(-1, 1)
    Y_onehot = ohe.fit_transform(Y_encoded_reshaped)

    # Save categories and encoder
    with open(f"categories_and_encoder_{classifier}.pkl", "wb") as file:
        pickle.dump((encoder.classes_, encoder), file)

    print(f"Category-Encoding Mapping:{classifier}")
    for label, encoded_label in zip(encoder.classes_, range(num_classes)):
        print(f"{label} -> {encoded_label}")

    return Y_onehot, num_classes
def main():
    X ={}
    Y={"scene":[],"device":[],"location":[]}
    os.chdir('D:/semesters/semester-4/Audio_files_train/Audio_files_train')
    epochs = 250
    embeddings_of = ['activation_1', 'activation_3', 'activation_5','flatten']
    stepmodel = model_loader(embeddings_of=embeddings_of)
    operation='sum'
    batch_num = 0
    models={}
    histories = {}
    for i in find_indexes(embeddings_of):
        X.update({f'{i}':[]})
    folderpath_train_val="./Audio_files"
    file_way=[]
    for file_paths_batch in get_file_paths_in_batches(folderpath_train_val, batch_size=32):
        file_way.extend(file_paths_batch)
        print('*'*25)
        datamaker(X,Y,stepmodel,operation,folder_path=folderpath_train_val,files=file_paths_batch, embeddings_of=embeddings_of) 
    for i in Y.keys():
        Y[i]=OnehotEncoding(Y[i],i)
    for i in find_indexes(embeddings_of):
        models.update({f'{i}':{}})
        histories.update({f'{i}':{}})
        for j in Y.keys():
            models[str(i)][j]=None
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Function to retrieve category names and encoder
    def get_category_names_and_encoder(classifier):
        with open(f"categories_and_encoder_{classifier}.pkl", "rb") as file:
            category_names, encoder = pickle.load(file)
            
        return category_names, encoder

    for i in find_indexes(embeddings_of):
        for j in Y.keys():
            if j != 'device':
                X_train, X_test, y_train, y_test = train_test_split(np.vstack(X[str(i)]), Y[j][0], test_size=0.2, random_state=42)
                print(f"calssification:{j} and model:{i}")
                models[str(i)][str(j)] = model_maker(X_train.shape[1], num_classes=Y[j][0].shape[1])
                histories[str(i)][str(j)] = models[str(i)][str(j)].fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

                # Get category names and encoder
                category_names, encoder = get_category_names_and_encoder(j)

                # Compute predictions on test set
                y_pred = models[str(i)][str(j)].predict(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_true = np.argmax(y_test, axis=1)

                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)

                # Plot confusion matrix with category names
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=category_names, yticklabels=category_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix for {j} (Model: {i})')
                plt.show()

            else:
                from collections import Counter
                labels = np.argmax(Y[j][0], axis=1)  # Calculate class frequencies
                class_counts = Counter(labels)
                total_samples = sum(class_counts.values())

                # Calculate class weights as inverse of class frequencies
                class_weights = {cls: total_samples / (class_counts[cls] * len(class_counts)) for cls in range(len(class_counts))}
                weights_array = np.array(list(class_weights.values()), dtype=np.float32)

                X_train, X_test, y_train, y_test = train_test_split(np.vstack(X[str(i)]), Y[j][0], test_size=0.2, random_state=42)

                new_lst = [np.sum(y_train[:, i]) for i in range(len(y_train[0]))]
                class_distribution(new_lst)

                new_lst = [np.sum(y_train[:, i]) for i in range(len(y_train[0]))]
                class_distribution(new_lst)

                models[str(i)][str(j)] = model_maker(X_train.shape[1], num_classes=Y[j][0].shape[1], class_weights=class_weights)
                histories[str(i)][str(j)] = models[str(i)][str(j)].fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

                # Get category names and encoder
                category_names, encoder = get_category_names_and_encoder(j)

                # Compute predictions on test set
                y_pred = models[str(i)][str(j)].predict(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_true = np.argmax(y_test, axis=1)

                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)

                # Plot confusion matrix with category names
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=category_names, yticklabels=category_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix for {j} (Model: {i})')
                plt.show()
    folderpath_train_val="./Test"
    X_test={}
    Y_test={"scene":[],"device":[],"location":[]}
    for i in find_indexes(embeddings_of):
        X_test.update({f'{i}':[]})
    for file_paths_batch in get_file_paths_in_batches(folderpath_train_val, batch_size=32):
        datamaker(X_test,Y_test,stepmodel,operation,folder_path=folderpath_train_val,files=file_paths_batch, embeddings_of=embeddings_of) 
    for i in Y_test.keys():
        Y_test[i]=OnehotEncoding(Y_test[i],i)
    from sklearn.metrics import accuracy_score, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    for i in find_indexes(embeddings_of):
        for j in Y.keys():
            if j != 'device':
                category_names, encoder = get_category_names_and_encoder(j)

                # Convert the list to a numpy array
                X_test_data = np.array(X_test[str(i)])
                y_test_data = Y_test[j][0]

                # Compute predictions on test set
                y_pred = models[str(i)][str(j)].predict(X_test_data)
                y_pred = np.argmax(y_pred, axis=1)
                y_true = np.argmax(y_test_data, axis=1)

                # Compute and print accuracy
                accuracy = accuracy_score(y_true, y_pred)
                print(f"Accuracy for {j} (Model: {i}): {accuracy:.4f}")

                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)

                # Plot confusion matrix with category names
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=category_names, yticklabels=category_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix for {j} (Model: {i})')
                plt.show()

                # Plot validation confusion matrix
            
            else:
                from collections import Counter
                labels = np.argmax(Y[j][0], axis=1)  # Calculate class frequencies

                # Convert the list to a numpy array
                X_test_data = np.array(X_test[str(i)])
                y_test_data = Y_test[j][0]

                y_pred = models[str(i)][str(j)].predict(X_test_data)
                y_pred = np.argmax(y_pred, axis=1)
                y_true = np.argmax(y_test_data, axis=1)

                # Compute and print accuracy
                accuracy = accuracy_score(y_true, y_pred)
                print(f"Accuracy for {j} (Model: {i}): {accuracy:.4f}")

                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)

                # Plot confusion matrix with category names
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=category_names, yticklabels=category_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix for {j} (Model: {i})')
                plt.show()

                # Plot validation confusion matrix
    def plot_losses(histories):
        for model_id, dict_ in histories.items():
            for j, history in dict_.items():
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title(f'Loss for Model {model_id} AND for classification {j}')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
        
                plt.subplot(1, 2, 2)
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.title(f'Loss for Model {model_id} AND for classification {j}')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.tight_layout()
                plt.show()
    plot_losses(histories)
    #os.chdir('../model')
    #for i in models.keys():
    #    for j in models[i].keys():
    #        models[str(i)][str(j)].save(f"classifier_maam_DCASE_SUM_MAX_{i}_class{j}.h5")