# -*- coding: utf-8 -*-
keep=['b','s1','s2','a','c','s3']
import json
def load_embeddings_from_json(file_path,device=False):
    with open(file_path, 'r') as file:
        embeddings_data = json.load(file)
        if(device==True):
            embeddings_data_copy=embeddings_data.copy()
            for i in embeddings_data.keys():
                if i.split('-')[4].split('.')[0] not in keep:
                   del embeddings_data_copy[i]
            return embeddings_data_copy
    return embeddings_data

import random

def shuffle_dict(dictionary):
    """
    Randomly shuffles the key-value pairs in the given dictionary.
    """
    keys = list(dictionary.keys())
    random.shuffle(keys)
    shuffled_dict = {k: dictionary[k] for k in keys}
    return shuffled_dict

layers=[7,14,21,28]
classifiers=['scene','device','location']

X={}
Y={}
for i in layers:
  X[str(i)]=shuffle_dict(load_embeddings_from_json(fr'./embeddings_basic/X_{i}'))
for i in ['device','location','scene']:
  Y[i]=load_embeddings_from_json(fr'./embeddings_basic/Y_{i}.json')
X_test={}
Y_test={}
for i in layers:
  X_test[str(i)]=shuffle_dict(load_embeddings_from_json(fr'./embeddings_basic/X_test_{i}.json'))
for i in ['device','location','scene']:
  Y_test[i]=load_embeddings_from_json(fr'./embeddings_basic/Y_test_{i}.json')
  if i=='device':
      Y_test[i]=load_embeddings_from_json(fr'./embeddings_basic/Y_test_{i}.json',device=True)
import requests
import soundfile as sf
import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
class DenseNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # Apply Xavier initialization
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE as openTSNE

def plot_tsne(embeddings, labels, classes_to_plot,j,is_aux=False,is_train=False):
    """
    Applies t-SNE to the given embeddings and plots the specified classes in a 2D scatter plot.
    
    Args:
        embeddings (torch.Tensor): Embeddings to be plotted.
        labels (torch.Tensor): One-hot encoded labels corresponding to the embeddings.
        classes_to_plot (list): List of classes to plot (represented by their indices in the one-hot encoding).
        is_train (bool, optional): If True, the plot will be labeled as "Train Embeddings". Default is False.
    """
    # Convert tensors to numpy arrays
    embeddings = embeddings.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = openTSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit(embeddings)
    
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    
    # Plot the specified classes
    for class_idx in classes_to_plot:
        class_mask = labels[:, class_idx] == 1
        plt.scatter(embeddings_2d[class_mask, 0], embeddings_2d[class_mask, 1], label=f"Class {class_idx}", s=10)
    
    # Add labels and title
    plt.legend()
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    if is_train:
        if is_aux:
            plt.title(f"t-SNE Visualization of:{j} with opl(Train Embeddings)")
        else:
            plt.title(f"t-SNE Visualization of:{j} (train Embeddings)")        
    else:
        if is_aux:    
            plt.title(f"t-SNE Visualization of:{j} with opl(Test Embeddings)")
        else:
            plt.title(f"t-SNE Visualization of:{j} (Test Embeddings)")
    plt.show()
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    _, true = torch.max(labels.data, 1)
    total = labels.size(0)
    correct = (predicted == true).sum().item()
    accuracy = correct / total
    return accuracy

import torch.nn as nn
class CombinedModel(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(CombinedModel, self).__init__()

        # Define the linear layer to convert embeddings to 704 size
        self.linear_layer = nn.Linear(embedding_size, 512)

        # Define the DenseNet
        self.dense_net = DenseNet(512, num_classes)

    def forward(self, embeddings,return_both=False):
        # Pass the embeddings through the linear layer
        linear_output = self.linear_layer(embeddings)

        # Pass the linear output through the DenseNet
        dense_net_output = self.dense_net(linear_output)
        if return_both:
            return dense_net_output, linear_output
        else:
            return dense_net_output

import matplotlib.pyplot as plt

X.keys()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.cuda as cuda
import numpy as np
# Load the embeddings and labels from the JSON file
# Create a PyTorch dataset
model_storer = {}

# Check if CUDA is available
if cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
best_test_acc_with_aux = []
best_test_acc_without_aux = []
gamma=0.5
num_epochs =250
t=4
for a in range(1,t+1):
    for i in layers:
        best_test_acc_with_aux_layer = []
        best_test_acc_without_aux_layer = []
        for j in classifiers:

            train_embeddings_with_opl = torch.tensor([])
            test_embeddings_with_opl = torch.tensor([])
            train_embeddings_without_opl = torch.tensor([])
            test_embeddings_without_opl = torch.tensor([])
            train_acc_with_aux = []
            test_acc_with_aux = []
            train_acc_without_aux = []
            test_acc_without_aux = []
            temp_X = X[str(i)]
            temp_Y = Y[j]
            a = []
            b = []
            for k in temp_X.keys():
                a.append(temp_X[k])
                b.append(temp_Y[k])
    
            X_tensor = torch.tensor(a)
            y_tensor = torch.tensor(b)
            num_classes=y_tensor.shape[1]
            # Move tensors to GPU
            X_tensor = X_tensor.to(device)
            y_tensor = y_tensor.to(device)
    
            train_set = TensorDataset(X_tensor, y_tensor)
    
            # Split the dataset into train and test sets
            temp_X_test = X_test[str(i)]
            temp_Y_test = Y_test[j]
            a_test = []
            b_test = []
            if j=='device':
                for k in temp_Y_test.keys():
                    a_test.append(temp_X_test[k])
                    b_test.append(temp_Y_test[k])
            else:    
                for k in temp_X_test.keys():
                    a_test.append(temp_X_test[k])
                    b_test.append(temp_Y_test[k])                
            X_tensor_test = torch.tensor(a_test)
            y_tensor_test = torch.tensor(b_test)
            # Move tensors to GPU
            X_tensor_test = X_tensor_test.to(device)
            y_tensor_test = y_tensor_test.to(device)
            test_set = TensorDataset(X_tensor_test, y_tensor_test)
            # Create data loaders for train and test sets
            batch_size = 256
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
            if j == 'scene':  # Initialize the model from scratch for the "scene" classifier
                model = CombinedModel(embedding_size=X_tensor.shape[1], num_classes=num_classes)
            else:  # Use the same linear layer as the "scene" classifier, but don't train it
                model = CombinedModel(embedding_size=X_tensor.shape[1], num_classes=num_classes)
                model.linear_layer.weight = scene_model.linear_layer.weight
                model.linear_layer.bias = scene_model.linear_layer.bias
                for param in model.linear_layer.parameters():
                    param.requires_grad = False
    
            # Move the model to GPU
            model = model.to(device)
    
            # Define the loss functions
            criterion = nn.CrossEntropyLoss()
            aux_criterion = OrthogonalProjectionLoss(gamma=gamma)
    
            # Define the optimizer
            optimizer = optim.Adam(model.parameters(),lr=5e-5)
    
            # Training loop
            for epoch in range(num_epochs):
              train_loss = 0.0
              train_acc = 0.0
              model.train()  # Set the model to training mode
              train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Train)")
              for k,(batch_embeddings, batch_labels) in enumerate(train_pbar):
                  batch_embeddings = batch_embeddings.to(device)
                  batch_labels = batch_labels.to(device)
    
                  # Forward pass
                  outputs, linear_outputs = model(batch_embeddings, return_both=True)
                  linear_outputs = linear_outputs.to(device)
                  # Calculate the cross-entropy loss
                  ce_loss = criterion(outputs, batch_labels)
    
                  # Calculate the custom loss
                  custom_loss = aux_criterion(linear_outputs, torch.argmax(batch_labels,1))
    
                  # Combine the losses
                  loss = ce_loss + 0.1 * custom_loss
                  # Backward pass
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
    
                  # Update the training loss and accuracy
                  train_loss += loss.item()
                  train_acc += compute_accuracy(outputs, batch_labels)
    
                  # Update the progress bar
                  train_pbar.set_postfix(loss=train_loss/(k+1), acc=train_acc/(k+1))
                  if epoch==num_epochs-1:
                      train_embeddings_with_opl = torch.cat((train_embeddings_with_opl, linear_outputs), dim=0)                  
              if epoch==num_epochs-1:
                  if j == "device":
                      classes_to_plot = [0,1,2,3,4,5]  # Specify the classes you want to plot
                      plot_tsne(train_embeddings_with_opl, y_tensor, classes_to_plot,j,is_aux=True,is_train=True)
                  elif j == "location":
                      classes_to_plot = [0,1,2,3,4,5]  # Specify the classes you want to plot
                      plot_tsne(train_embeddings_with_opl, y_tensor, classes_to_plot,j,is_aux=True,is_train=True)
                  else:
                      classes_to_plot = [0,1,2,3,4,5]  # Specify the classes you want to plot
                      plot_tsne(train_embeddings_with_opl, y_tensor, classes_to_plot,j,is_aux=True,is_train=True)
            # Calculate the average training loss and accuracy
              train_loss = train_loss / len(train_pbar)
              train_acc = train_acc / len(train_pbar)
    
            # Evaluation on the test set
              model.eval()  # Set the model to evaluation mode
              test_loss = 0.0
              test_acc = 0.0
              with torch.no_grad():  # Disable gradient calculation
                  test_pbar = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Test)")
                  for k,(batch_embeddings, batch_labels) in enumerate(test_pbar):
                      batch_embeddings = batch_embeddings.to(device)
                      batch_labels = batch_labels.to(device)
    
                      outputs,linear_outputs= model(batch_embeddings, return_both=True)
                      if j=="device":
                          batch_labels = batch_labels[:, :-3]
                      loss = criterion(outputs, batch_labels)
                      test_loss += loss.item()
                      test_acc += compute_accuracy(outputs, batch_labels)
                      # Update the progress bar
                      test_pbar.set_postfix(loss=test_loss/(k+1), acc=test_acc/(k+1))
                      if epoch==num_epochs-1:
                            test_embeddings_with_opl = torch.cat((test_embeddings_with_opl, linear_outputs), dim=0)
              if epoch==num_epochs-1:
                  if j == "device":
                      classes_to_plot =[0,1,2]   # Specify the classes you want to plot
                      plot_tsne(test_embeddings_with_opl, y_tensor_test[:,:-3], classes_to_plot,j,is_aux=True)
                  elif j == "location":
                      classes_to_plot = [0,1,2]    # Specify the classes you want to plot
                      plot_tsne(test_embeddings_with_opl, y_tensor_test, classes_to_plot,j,is_aux=True)
                  else:
                      classes_to_plot =[0,1,2]    # Specify the classes you want to plot
                      plot_tsne(test_embeddings_with_opl, y_tensor_test, classes_to_plot,j,is_aux=True)
              # Calculate the average test loss and accuracy
              test_loss = test_loss / len(test_pbar)
              test_acc = test_acc / len(test_pbar)
              train_acc_with_aux.append(train_acc)
              test_acc_with_aux.append(test_acc)
              # Print the loss and accuracy for the current epoch
              print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
            print("*"*40)
        # Train the model without auxiliary loss
            if j == 'scene':  # Initialize the model from scratch for the "scene" classifier
                model_wout_aux = CombinedModel(embedding_size=X_tensor.shape[1], num_classes=num_classes)
                criterion__wout_aux = nn.CrossEntropyLoss()
                optimizer_wout_aux =  optim.Adam(model_wout_aux.parameters(),lr=5e-5)
            else:  # Use the same linear layer as the "scene" classifier, but don't train it
                model_wout_aux = CombinedModel(embedding_size=X_tensor.shape[1], num_classes=num_classes)
                model_wout_aux.linear_layer.weight = scene_model_wout_aux.linear_layer.weight
                model_wout_aux.linear_layer.bias = scene_model_wout_aux.linear_layer.bias
                criterion__wout_aux = nn.CrossEntropyLoss()
                optimizer_wout_aux =  optim.Adam(model_wout_aux.parameters(),lr=5e-5)
                for param in model_wout_aux.linear_layer.parameters():
                    param.requires_grad = False
            model_wout_aux = model_wout_aux.to(device)
            for epoch in range(num_epochs):
                train_loss = 0.0
                train_acc = 0.0
                model_wout_aux.train()
                train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Train)")
                for k,(batch_embeddings, batch_labels) in enumerate(train_pbar):
                    batch_embeddings = batch_embeddings.to(device)
                    batch_labels = batch_labels.to(device)
    
                    outputs,linear_output= model_wout_aux(batch_embeddings, return_both=True)
                    loss_wout_aux = criterion__wout_aux(outputs, batch_labels)
                    optimizer_wout_aux.zero_grad()
                    loss_wout_aux.backward()
                    optimizer_wout_aux.step()
                    train_loss += loss_wout_aux.item()
                    train_acc += compute_accuracy(outputs, batch_labels)
                    train_pbar.set_postfix(loss=train_loss/(k+1), acc=train_acc/(k+1))
                    if epoch==num_epochs-1:
                        train_embeddings_without_opl=torch.cat((train_embeddings_without_opl, linear_output), dim=0)
                if epoch==num_epochs-1:
                    if j == "device":
                        classes_to_plot = [0,1,2]   # Specify the classes you want to plot
                        plot_tsne(train_embeddings_without_opl, y_tensor, classes_to_plot,j,is_train=True)
                    elif j == "location":
                        classes_to_plot =[0,1,2]    # Specify the classes you want to plot
                        plot_tsne(train_embeddings_without_opl, y_tensor, classes_to_plot,j,is_train=True)
                    else:
                        classes_to_plot = [0,1,2]    # Specify the classes you want to plot
                        plot_tsne(train_embeddings_without_opl, y_tensor, classes_to_plot,j,is_train=True)

                train_loss = train_loss / len(train_pbar)
                train_acc = train_acc / len(train_pbar)
    
                model_wout_aux.eval()
                test_loss = 0.0
                test_acc = 0.0
                with torch.no_grad():
                    test_pbar = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Test)")
                    for k,(batch_embeddings, batch_labels) in enumerate(test_pbar):
                        batch_embeddings = batch_embeddings.to(device)
                        batch_labels = batch_labels.to(device)
    
                        outputs,linear_output= model_wout_aux(batch_embeddings, return_both=True)
                        if j=="device":
                            batch_labels = batch_labels[:, :-3]
                        loss_wout_aux = criterion__wout_aux(outputs, batch_labels)
                        test_loss += loss_wout_aux.item() 
                        test_acc += compute_accuracy(outputs, batch_labels)
                        test_pbar.set_postfix(loss=test_loss/(k+1), acc=100 *test_acc/(k+1))
                        if epoch==num_epochs-1:
                            test_embeddings_without_opl=torch.cat((test_embeddings_without_opl, linear_output), dim=0)
                if epoch==num_epochs-1:
                    if j == "device":
                        classes_to_plot = [0,1,2]    # Specify the classes you want to plot
                        plot_tsne(test_embeddings_without_opl, y_tensor_test[:,:-3], classes_to_plot,j)
                    elif j == "location":
                        classes_to_plot = [0,1,2]   # Specify the classes you want to plot
                        plot_tsne(test_embeddings_without_opl, y_tensor_test, classes_to_plot,j)
                    else:
                        classes_to_plot = [0,1,2]    # Specify the classes you want to plot
                        plot_tsne(test_embeddings_without_opl, y_tensor_test, classes_to_plot,j)
                test_loss = test_loss / len(test_pbar)
                test_acc = test_acc / len(test_pbar)
    
                # Store the train and test accuracies without auxiliary loss
                train_acc_without_aux.append(train_acc)
                test_acc_without_aux.append(test_acc)
                
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
            import time
            import json

            # Convert NumPy arrays to Python lists
 # Plot the train and test accuracies
            import matplotlib.pyplot as plt
    
            plt.figure(figsize=(8, 6))
            epochs = range(1, num_epochs + 1)
            plt.plot(epochs, test_acc_with_aux, label='Test Accuracy (with aux)')
            plt.plot(epochs, test_acc_without_aux, label='Test Accuracy (without aux)')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'{j},{i}Test Accuracy with and without Auxiliary Loss')
            plt.legend()
            plt.tight_layout()
            plt.show()
            best_test_acc_with_aux.append(max(test_acc_with_aux))
            best_test_acc_without_aux.append(max(test_acc_without_aux))
            if j == 'scene':
              scene_model=model
              scene_model_wout_aux=model_wout_aux
    
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgb
    
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgb
    
    # Assuming you have the following lists:
    best_test_acc_with_aux =  np.array(best_test_acc_with_aux).reshape(-1,3)
    best_test_acc_without_aux = np.array(best_test_acc_without_aux).reshape(-1,3)
    layers = layers
    classifiers = ['scene', 'device', 'location']
    
    # Create a list of indices for the x-axis
    x = np.arange(len(layers))
    
    # Set the width of the bars
    bar_width = 0.15
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for the bars
    colors = ['r', 'g', 'b']
    shaded_colors = [np.clip(to_rgb(color) + np.array([0.2, 0.2, 0.2]), 0, 1) for color in colors]
    once=True
    # Loop through layers and plot the bars
    for i, layer in enumerate(layers):
        if once:
          offsets = np.arange(len(classifiers)) - len(classifiers) / 2
    
          # Plot the bars for with auxiliary loss
          for j, (acc, offset) in enumerate(zip(best_test_acc_with_aux[i], offsets)):
              ax.bar(x[i] + offset * bar_width * 2, acc, bar_width, label=f'Layer {layer}, Classifier {classifiers[j]} (with aux)', color=colors[j], edgecolor='black', hatch='///')
    
          # Plot the bars for without auxiliary loss
          for j, (acc, offset) in enumerate(zip(best_test_acc_without_aux[i], offsets)):
              ax.bar(x[i] + (offset+0.5) * bar_width * 2, acc, bar_width, label=f'Layer {layer}, Classifier {classifiers[j]} (without aux)', color=shaded_colors[j], edgecolor='black')
              once=False
        else:
          offsets = np.arange(len(classifiers)) - len(classifiers) / 2
    
          # Plot the bars for with auxiliary loss
          for j, (acc, offset) in enumerate(zip(best_test_acc_with_aux[i], offsets)):
              ax.bar(x[i] + offset * bar_width * 2, acc, bar_width, color=colors[j], edgecolor='black', hatch='///')
    
          # Plot the bars for without auxiliary loss
          for j, (acc, offset) in enumerate(zip(best_test_acc_without_aux[i], offsets)):
              ax.bar(x[i] + (offset+0.5) * bar_width * 2, acc, bar_width, color=shaded_colors[j], edgecolor='black')
    # Add labels and title
    ax.set_xlabel('Layer')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Best Test Accuracy with and without Auxiliary Loss')
    ax.set_xticks(x)  # Set the x-ticks to the layer indices
    ax.set_xticklabels(layers)
    
    # Add a legend
    ax.legend()
    
    # Adjust the spacing between bars
    plt.subplots_adjust(bottom=0.2)
    
    # Show the plot
    plt.show()
    
