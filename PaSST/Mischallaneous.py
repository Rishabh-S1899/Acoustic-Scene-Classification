import ast 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score


def train_ready(data,col_name):
    new_list=[]
    for i in range(len(data)):
        temp_list=[]
        x=data[col_name][i]
        tensor_string = x

# Extract the numbers part of the string
        numbers_string = tensor_string.split("[[")[1].split("]]")[0]

# Parse the string into a Python list
        tensor_list = ast.literal_eval("[" + numbers_string + "]")

# Convert the list to a numpy array
        numpy_array = np.array(tensor_list)
        list_data=list(numpy_array)
        
        new_list.append(list_data)
    return new_list


def class_distribution(data_list,kk='train'):
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
    print(f"Per-class distribution for {kk} data:")
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

def one_hot(lst,run_mode='scene'):
    #one hot encoding for the labels
    y=lst[run_mode].tolist()
    label_array=np.array(y).reshape(-1, 1)
    enc=OneHotEncoder()
    encoded_lst = enc.fit_transform(label_array).toarray()
    return encoded_lst

