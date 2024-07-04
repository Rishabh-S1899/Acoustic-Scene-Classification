import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import openl3
import matplotlib.pyplot as plt
import os
import soundfile
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
    
# Define your custom layer or operation

def grad_cam(model, img_tensor,
             layer_names=["audio_embedding_layer",'conv2d_5','conv2d_3','conv2d_1'], label_name=None,
             category_id=None):
    """Get a heatmap by Grad-CAM.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    heatmaps=[]
    for layer_name in layer_names:
        conv_layer = model.get_layer(layer_name)
        heatmap_model = Model(model.inputs, [conv_layer.output, model.output])
    
        with tf.GradientTape() as gtape:
            conv_output, predictions = heatmap_model(img_tensor)
            if category_id is None:
                category_id = np.argmax(predictions[0])
            if label_name is not None:
                print(label_name[category_id])
            print(predictions)
            output = predictions[:, category_id]
            grads = gtape.gradient(output, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        print(heatmap.shape)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat
        heatmaps.append(np.squeeze(heatmap))
    return heatmaps
def grad_cam_plus(model, img_tensor,
                  layer_names=["audio_embedding_layer",'conv2d_6','conv2d_5','conv2d_4'], label_name=None,
                  category_id=None):
    """Get a heatmap by Grad-CAM++.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    heatmaps=[]
    for layer_name in layer_names:
        conv_layer = model.get_layer(layer_name)
        heatmap_model = Model(model.inputs, [conv_layer.output, model.output])
        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    conv_output, predictions = heatmap_model(img_tensor)
                    if category_id is None:
                        category_id = np.argmax(predictions[0])
                    if label_name is not None:
                        print(label_name[category_id])
                    output = predictions[:, category_id]
                    conv_first_grad = gtape3.gradient(output, conv_output)
                conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
            conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)
        
        global_sum = np.sum(conv_output, axis=(1, 2))
        alpha_num = conv_second_grad
        alpha_denom = conv_second_grad*2.0 + conv_third_grad*global_sum[:,np.newaxis,np.newaxis,:]
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)
    
        alphas = alpha_num/alpha_denom
        alpha_normalization_constant = np.sum(alphas, axis=(1,2))
        alphas /= alpha_normalization_constant[:,np.newaxis,np.newaxis,:]
        weights = np.maximum(conv_first_grad, 0.0)
    
        deep_linearization_weights = np.sum(weights*alphas, axis=(1,2))
        grad_cam_map = np.sum(deep_linearization_weights[:,np.newaxis,np.newaxis,:]*conv_output, axis=3)
        heatmap = np.maximum(grad_cam_map, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat
        heatmaps.append(heatmap)
    return heatmaps
def find_indexes(subset):
    superset = ['melspectrogram', 'batch_normalization', 'conv2d', 'batch_normalization_1', 'activation', 'conv2d_1', 'batch_normalization_2', 'activation_1', 'max_pooling2d', 'conv2d_2', 'batch_normalization_3', 'activation_2', 'conv2d_3', 'batch_normalization_4', 'activation_3', 'max_pooling2d_1', 'conv2d_4', 'batch_normalization_5', 'activation_4', 'conv2d_5', 'batch_normalization_6', 'activation_5', 'max_pooling2d_2', 'conv2d_6', 'batch_normalization_7', 'activation_6', 'audio_embedding_layer', 'max_pooling2d_3', 'flatten']
    indexes = []
    for element in subset:
            index = superset.index(element)
            indexes.append(index)
    del superset
    return indexes

def datamaker(f, hop_size=1):
    model = openl3.models.load_audio_embedding_model(input_repr, content_type, embedding_size)
    outs = []
    y = model.input
    layers = model.layers[1:]
    for layer in layers:
        y = layer(y)
        outs.append(y)
    models=type(model)(model.inputs, outs)
    del model,outs
    y, sr = soundfile.read(f)
    batches = openl3.core._preprocess_audio_batch(y, sr, hop_size=hop_size)
    return batches,models.predict(batches)[0]

def bilinear_interpolation(a):
    import numpy as np
    from scipy.ndimage import zoom
    
    output_array=[]
    for i in range(len(a)):
        input_array=a[i]
        # Calculate the zoom factors for each dimension
        zoom_x = 256 / a.shape[1]
        zoom_y = 199 / a.shape[2]
        
        # Bilinearly interpolate the input array to the desired size
        output_array.append(zoom(input_array, zoom=(zoom_x, zoom_y), order=1))
    return output_array

@keras.saving.register_keras_serializable()
class BatchReshape(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchReshape, self).__init__(**kwargs)

    def call(self, inputs):
        shape = tf.shape(inputs)
        num_batches = shape[0]
        feature_size = shape[1]
        return tf.reshape(inputs, (1, feature_size * 11))
def plotter(a):
    import numpy as np
    import matplotlib.pyplot as plt
    # Generate a random tensor of shape (32, 24)
    for i in range(len(a)):
        tensor = a[i]
        # Plot the tensor
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(tensor, cmap='inferno')
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        # Add labels
        ax.set_title(f'Tensor Visualization of part{i}')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        # Show the plot
        plt.show()

def model_combiner(new_directory,model_path):
    pretrained_model = tf.keras.models.load_model(model_path)
    layer_names = [layer.name for layer in pretrained_model.layers]
    weights_pretrained=[]
    for layer_name in layer_names:
        layer = pretrained_model.get_layer(layer_name)
        weights_pretrained.append(layer.get_weights())
    if(len(layer_names)==3):
        from tensorflow.keras.layers import Input, Dense

        inputs = Input(shape=(5632,), name='input_layer')
        x = Dense(512, activation='relu', name='dense_layer',weights=weights_pretrained[0])(inputs)
        sequential_2 = Model(inputs=inputs, outputs=x, name='sequential_2')
        
        sequential_3 = Sequential([
            Dense(128, activation='relu', name=layer_names[1]+"new", weights=pretrained_model.layers[1].get_weights()),
            # Add more layers if needed
        ], name='sequential_3')
        
        sequential_4 = Sequential([
            Dense(pretrained_model.layers[2].output_shape[1], activation='softmax', name=layer_names[2]+"new", weights=pretrained_model.layers[2].get_weights()),
            # Add more layers if needed
        ], name='sequential_4')
        
        audio_embedding_output = openl3_model.output
        reshaped_output = BatchReshape()(audio_embedding_output)
        sequential_2_output = sequential_2(reshaped_output)
        sequential_3_output = sequential_3(sequential_2_output)
        sequential_4_output = sequential_4(sequential_3_output)
        
        combined_model = Model(inputs=openl3_model.input, outputs=sequential_4_output)
        
    if(len(layer_names)==2):
        sequential_2 = Sequential([
        Dense(256, activation='relu', input_shape=(512,), name=layer_names[0], weights=pretrained_model.layers[0].get_weights()),
        # Add more layers if needed
        ], name='sequential_2')

        sequential_3 = Sequential([
        Dense(pretrained_model.layers[1].output_shape[1], activation='softmax', name=layer_names[1], weights=pretrained_model.layers[1].get_weights()),
        # Add more layers if needed
        ], name='sequential_3')

        audio_embedding_output = openl3_model.output
        sequential_2_output = sequential_2(audio_embedding_output)
        sequential_3_output = sequential_3(sequential_2_output)
        combined_model = Model(inputs=openl3_model.input, outputs=sequential_3_output)
    #Compile the combined model
    combined_model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy')
    model_save_name="attached_"+os.path.basename(model_path)
    model_save_path=os.path.join(new_directory,model_save_name)
    combined_model.save(model_save_path[:len(model_save_path)-2]+"keras")
    print(combined_model.summary())
    print("Model saved successfully at:", model_save_path)
    return combined_model
def concatinator(primary_tensor,secondary_tensor):
    primary_joined = np.concatenate(primary_tensor, axis=1)  # Shape: (256, 199*11)
    secondary_joined = np.concatenate(secondary_tensor, axis=1)  # Shape: (256, 199*11)
    
    # 2. Plot the primary tensor
    fig, ax = plt.subplots(figsize=(20,12))
    im = ax.imshow(primary_joined, cmap='viridis')
    ax.set_title('Primary Tensor')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    #plt.colorbar(im, ax=ax)
    plt.show()
    # 3. Plot the secondary tensor on top with lower opacity
    fig, ax = plt.subplots(figsize=(20, 12))
    im_overlay = ax.imshow(secondary_joined, cmap='viridis', alpha=0.5)  # Set the desired opacity here
    ax.set_title('Secondary Tensor Overlay')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    #plt.colorbar(im, ax=ax)
    
    plt.show()
model_save_path_folder = r"D:\semesters\semester-4\Audio_files_train\model"
new_directory = os.path.join(model_save_path_folder, "joined")
input_repr = "mel256"
content_type = "music"
embedding_size = 512
openl3_model = openl3.models.load_audio_embedding_model(input_repr, content_type, embedding_size)
model=model_combiner(new_directory,os.path.join(model_save_path_folder,'classifier_maam_DCASE_SUM_MAX_28_classscene.h5'))
img,spectrogram=datamaker('D:/semesters/semester-4/Audio_files_train/Audio_files/airport-barcelona-1-18-a.wav')
heatmaps=grad_cam(model,img)
for heatmap in heatmaps:
    interpolated_heatmap=bilinear_interpolation(heatmap)
    print(model.predict(img))
    concatinator(spectrogram,interpolated_heatmap)