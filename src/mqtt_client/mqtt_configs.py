import enum
from dataclasses import dataclass

from paho.mqtt import client as mqtt

from src.commons import InputData


class Topics(enum.Enum):
    registration = "devices/"
    device_inference = "device_01/model_inference"
    device_inference_result = "device_01/model_inference_result"
    end_computation = "device_01/end_computation"


@dataclass
class MqttClientConfig:
    broker_url: str = "hostname.local"
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

image_array = InputData().image_array.tolist()

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
