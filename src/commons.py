import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class OffloadingDataFiles:
    data_file_path_device: str = "../device_inference_times.json"
    data_file_path_edge: str = "../edge_inference_times.json"
    data_file_path_sizes: str = "../layer_sizes.json"
    evaluation_file_path: str = "../evaluations/evaluations.csv"

class InputDataFiles:
    input_data_file_path = "../models/test/test_model/pred_data/pred_data_1.png" # Path to test image

class InputData:
    def __init__(self, image_path=InputDataFiles.input_data_file_path, color_mode="grayscale", target_size=(96, 96)): # Model input configuration
        input_image = load_img(image_path, color_mode=color_mode, target_size=target_size)
        image_array = img_to_array(input_image)
        self.image_array = np.array([image_array])