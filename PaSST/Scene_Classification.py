#Run this script for scene classification on embeddings using PaSST
import matplotlib.pyplot as plt
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import ast 
from Mischallaneous import train_ready,one_hot
from Classification_model import SceneClassifier
import tqdm 
import copy

run_mode='scene'

#Importing Data
data=pd.read_csv("your_train_embedding_csv_file_path")
test_data=pd.read_csv("your_test__embedding_csv_file_path")
df=pd.DataFrame() #Empty dataframe to store predictions 
df['file_name']=test_data['file_name']
acc_list=[]
loss_list=[]
number_of_transformer_blocks=12
#Check if CUDA device or not 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
number_of_classes=len(list(np.unique(np.array(data[run_mode]))))

input_dim = 768
output_dim = number_of_classes

#One Hot Encoding train and Test Labels
y_one_hot_encoded=one_hot(data[run_mode],run_mode) #set the run mode accordingly, default set to scene
y_test_one_hot_encoded=one_hot(test_data[run_mode],run_mode)

class_names = list(np.unique(test_data[run_mode].tolist()))

criterion = nn.CrossEntropyLoss()
model = SceneClassifier(input_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
n_epochs = 200
batch_size = 64

train_loss_tracker=[]
train_acc_tracker=[]
test_loss_tracker=[]
test_acc_tracker=[]
y_pred_all_block_list=[]
for i in range(1,number_of_transformer_blocks):
    final_X_train=train_ready(data,f'embedding{i}')

    X_train, X_val, y_train, y_val = train_test_split(final_X_train, y_one_hot_encoded, test_size=0.2, random_state=42, stratify=y_one_hot_encoded)
    model = SceneClassifier(input_dim, output_dim).to(device)
    batches_per_epoch = len(X_train) // batch_size

    X_train_tensor=torch.tensor(X_train,dtype=torch.float32)
    X_Val_tensor=torch.tensor(X_val,dtype=torch.float32)
    y_train_tensor=torch.tensor(y_train,dtype=torch.float32)
    y_val_tensor=torch.tensor(y_val,dtype=torch.float32)
    best_loss =  np.inf  # init to negative infinity
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    # training loop
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                X_batch = X_train_tensor[start:start + batch_size].to(device)
                y_batch = y_train_tensor[start:start + batch_size].to(device)

                # forward pass
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # update weights
                optimizer.step()

                # compute and store metrics
                acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(loss=float(loss), acc=float(acc))

        # set model in evaluation mode and run through the test set
        model.eval()
        with torch.no_grad():  # Disable gradient calculation for efficiency
            X_test_batch = X_Val_tensor.to(device)  # Move test data to device
            y_val_batch=y_val_tensor.to(device)
            y_pred = model(X_test_batch).to(device)
            ce = criterion(y_pred, y_val_batch)
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val_batch, 1)).float().mean()

        ce = float(ce)
        acc = float(acc)
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        
        if ce < best_loss:
            best_loss=ce
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc * 100:.1f}%")
    train_loss_tracker.append(train_loss_hist)
    test_loss_tracker.append(test_loss_hist)
    train_acc_tracker.append(train_acc_hist)
    test_acc_tracker.append(test_acc_hist)
    model.load_state_dict(best_weights)
    torch.save(model, fr'\Classification\Models\{run_mode}_{i}.pth')
    model=torch.load(fr'\Classification\Models\{run_mode}_{i}.pth')
    model.to(device)
    X_test=train_ready(test_data,f'embedding{i}')
    X_test_tensors = [torch.tensor(feature, dtype=torch.float32, device=device) for feature in X_test]
    # Run inference and get predictions
    y_pred = []
    with torch.no_grad():
        for feature in X_test_tensors:
            output = model(feature.unsqueeze(0))
            _, predicted = torch.max(output.data, 1)
            predicted_class = class_names[predicted.item()]
            y_pred.append(predicted_class)
    df[f'embedding{i}']=y_pred
    y_pred_all_block_list.append(y_pred)    
    print(f"Block {i} Completed")

print("Here is the accuracy list for the blocks: ")
for i in range(len(acc_list)):
    print(f"Block {i} accuracy: {acc_list[i]}\n")

num_histories = len(train_loss_tracker)  # Assuming you have a list of train_loss_hist and test_loss_hist
fig, axs = plt.subplots(2, num_histories, figsize=(20, 8))

for i, (train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist, run_mode) in enumerate(zip(train_loss_tracker, test_loss_tracker, train_acc_tracker, test_acc_tracker, run_mode)):
    block_number = i + 1  # Iterative block number
    # Plot loss histories
    ax = axs[0, i]
    ax.plot(train_loss_hist, label="train")
    ax.plot(test_loss_hist, label="test")
    ax.set_title(f"Loss for {run_mode} Classification on Embedding after Block: {block_number}")
    ax.set_xlabel("epochs")
    ax.set_ylabel("cross entropy")
    ax.legend()

    # Plot accuracy histories
    ax = axs[1, i]
    ax.plot(train_acc_hist, label="train")
    ax.plot(test_acc_hist, label="test")
    ax.set_title(f"Accuracy for {run_mode} Classification on Embedding after Block: {block_number}")
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracy")
    ax.legend()

plt.tight_layout()
plt.show()


num_matrices = len(y_pred)  # Number of confusion matrices
fig, axes = plt.subplots(1, num_matrices, figsize=(15, 5))

for i, y_pred_matrix in enumerate(y_pred):
    ax = axes[i]
    conf_matrix = confusion_matrix(test_data[run_mode].tolist(), y_pred_matrix, labels=class_names)
    accuracy = accuracy_score(test_data[run_mode].tolist(), y_pred_matrix)

    ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix {i+1} (Accuracy: {accuracy:.2f})')

    # Add numerical values to the confusion matrix
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, conf_matrix[i, j], ha="center", va="center",
                    color="white" if conf_matrix[i, j] > (conf_matrix.max() / 2) else "black")

plt.colorbar(ax=axes.ravel().tolist(), fraction=0.05, pad=0.05)  # Add a colorbar for all subplots
plt.tight_layout()
plt.show()