import enum
from dataclasses import dataclass

from paho.mqtt import client as mqtt

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from src.commons import InputDataFiles


class Topics(enum.Enum):
    registration = "devices/"
    device_inference = "device_01/model_inference"
    device_inference_result = "device_01/model_inference_result"
    end_computation = "device_01/end_computation"


@dataclass
class MqttClientConfig:
    broker_url: str = "FABIO-PC.local"
    broker_port: int = 1883
    client_id: str = "edge"
    subscribe_topics: list = (
        Topics.registration.value,
        Topics.device_inference.value,
        Topics.device_inference_result.value,
        Topics.end_computation.value
    )
    protocol: mqtt.MQTTv311 = mqtt.MQTTv311
    ntp_server: str = "time.google.com"

def image_to_np_array(image_path: str = InputDataFiles.input_data_file_path):
    input_image = load_img(image_path, color_mode="grayscale", target_size=(96, 96))
    image_array = img_to_array(input_image)
    image_array = np.array([image_array])
    return image_array

image_array = image_to_np_array().tolist()

@dataclass
class DefaultMessages:
    ask_for_inference_msg = {
        "device_id": "edge",
        "message_id": "edge",
        "timestamp": None,
        "message_content": "AskInference",
        "offloading_layer_index": None,
        "input_data": image_array
    }

    end_computation_msg = {
        "device_id": "edge",
        "message_id": "edge",
        "timestamp": None,
        "message_content": "EndComputation"
    }
