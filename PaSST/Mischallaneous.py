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
def load_audio_to_tensor(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=32000)
    y_tensor = torch.from_numpy(y_resampled).float()
    y_tensor = y_tensor.unsqueeze(0)
    return y_tensor.cuda() 

def generate_embeddings_and_save_to_csv(folder_path, model, device):
    """Generates embeddings for files in a folder and saves them to a CSV file.

    Args:
        folder_path (str): Path to the folder containing the files.
        model: The embedding generation model.
        device: The device to use for model processing (e.g., "cuda" or "cpu").
    """
    save_path="Classification/Embedding_files/PaSST_Embeddings/hear21_embeddings_blocks.csv"
    embeddings = []
    with open(fr"Classification/Embedding_files/PaSST_Embeddings/hear21_embeddings_blocks.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "embedding1", "embedding2", "embedding3", "embedding4", "embedding5", "embedding6", "embedding7", "embedding8", "embedding9", "embedding10", "embedding11", "embedding12","pre_logits"])

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Load input data from file_path (replace with your specific logic)
                val = load_audio_to_tensor(file_path)  # Replace with your loading function

                # Move model and input to CUDA device
                model = model.to(device)
                val = val.to(device)

                with torch.no_grad():
                    # Generate embedding using the model
                    # embedding = model(val).cpu().detach().numpy().tolist()
                    embedding=model(val)

                writer.writerow([filename]+embedding[1]+[embedding[0]])
                print(f"file {filename} encoded")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return save_path
            
def extract_info(filename):
    filename = filename.split('.')[0] #Remove.wav extension
    parts = filename.split('-')
    return {'scene': parts[0], 'location': parts[1], 'device': parts[-1]}
