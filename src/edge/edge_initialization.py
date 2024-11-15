import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

from src.commons import OffloadingDataFiles
from src.commons import InputDataFiles
from src.models.model_manager import ModelManager


if __name__ == "__main__":
    
    def image_to_np_array(image_path: str = InputDataFiles.input_data_file_path):
        input_image = load_img(image_path, color_mode="grayscale", target_size=(96, 96))
        image_array = img_to_array(input_image)
        image_array = np.array([image_array])
        return image_array

    # original array (grayscale image with shape (1, 96, 96, 1))
    image_array = image_to_np_array()

    # check the shape and dtype
    print(image_array.shape)  # Should print (1, 96, 96, 1)
    print(image_array.dtype)  # Should print float32

    # convert the NumPy array to a string representation for writing to a file
    image_str = np.array2string(image_array)

    # load the model and make predictions
    model_manager = ModelManager()
    model_manager.load_model()

    # set the layers to use
    predictions = {} 
    layer_sizes = {}
    start_layer_index = 0

    # set end_layer_index to the number of layers if not provided
    end_layer_index = len(model_manager.model.layers)

    # adjust model start layer index (offset is applied if the start index is 0 and the 0th layer is an InputLayer)
    start_layer_offset = 1 if isinstance(model_manager.model.layers[start_layer_index], tf.keras.layers.InputLayer) else 0
    start_layer_index = start_layer_index+start_layer_offset if start_layer_index == 0 else start_layer_index

    # set the layers to use
    layers_to_use = model_manager.model.layers[start_layer_index:end_layer_index]

    # loop through the layers and make predictions
    input_data = image_array

    for layer_index, layer in enumerate(layers_to_use, start=start_layer_index):
        if layer_index == start_layer_offset:   # if it's the first layer
            prediction_data = input_data
        else:
            # Get the previous layers' output tensor
            inbound_node = layer._inbound_nodes[0]
            if isinstance(inbound_node.inbound_layers, list):
                prediction_data = []
                for inbound_layer in inbound_node.inbound_layers:
                    prediction_data.append(predictions[inbound_layer])
            else:
                prediction_data = predictions[inbound_node.inbound_layers]

        prediction = model_manager.predict_single_layer(layer_index, start_layer_offset, prediction_data)
        layer_sizes[layer_index-start_layer_offset] = float(model_manager.get_layer_size_in_bytes(layer, prediction))
        predictions[layer] = prediction

    # save the inference times to a file
    model_manager.save_inference_times()

    # save the layer sizes to a file
    with open(OffloadingDataFiles.data_file_path_sizes, "w") as f:
        json.dump(layer_sizes, f, indent=4)
