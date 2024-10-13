import tensorflow as tf
from tensorflow.keras.models import Model

from model_manager_config import ModelManagerConfig


def load_h5(name: str, dir_path: str) -> Model:
    return tf.keras.models.load_model(f'{dir_path}/{name}.h5')

def save_keras(name: str, model: Model, dir_path: str) -> None:
    model.save(f'{dir_path}/{name}.keras')

def convert_h5_to_keras(name: str, dir_path: str) -> None:
    model = load_h5(name=name, dir_path=dir_path)
    save_keras(name=name, model=model, dir_path=dir_path)


if __name__ == "__main__":
    main_folder = "test/" + ModelManagerConfig.MODEL_DIR_PATH
    convert_h5_to_keras(name="test_model", dir_path=main_folder)
