class OffloadingDataFiles:
    data_file_path_device: str = "../device_inference_times.json"
    data_file_path_edge: str = "../edge_inference_times.json"
    data_file_path_sizes: str = "../layer_sizes.json"
    evaluation_file_path: str = "../evaluations/evaluations.csv"

class InputDataFiles:
    input_data_file_path = "../models/test/test_model/pred_data/pred_data_1.png"