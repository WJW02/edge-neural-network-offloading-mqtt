# Edge Server Neural Network Offloading with MQTT
This project implements an Edge Server that leverages an offloading algorithm to manage computational load between IoT devices and the server. The Edge Server evaluates the MQTT requests from connected devices and decides whether to offload the computation or let the device handle it locally based on factors such as network bandwidth, server load, and device capabilities.
## Configuration
```sh
git clone https://github.com/WJW02/edge-neural-network-offloading-mqtt.git
cd edge-neural-network-offloading-mqtt/
```
- Clone and move in the repository

```sh
pyenv install 3.10.12
```
- This is the python version used to test the project
- Newer versions of python like 3.12.X don't support the tensorflow version used in the project

```sh
pyenv global 3.10.12
```
- Switch to python 3.10.12
- You can switch back after configuration by running `pyenv system global`

```sh
python3 -m venv venv
```
- Create virtual environment

```sh
source venv/bin/activate
```
- Activate virtual environment

```sh
pip3 install .
```
- Install project dependencies

```sh
cd $(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo /absolute/path/to/project > project.pth
```
- Configure the absolute path to project (e.g. `~/Documents/edge-neural-network-offloading-mqtt/`)
## Model setup
- Save your keras model as `test_model.h5` in `src/models/test/test_model/`
- Save your test images in `src/models/test/test_model/pred_data/`
- Configure `InputDataFiles` and `InputData` in `src/commons.py` as needed
## Mqtt server setup
- Configure `MqttClientConfig` in  `src/mqtt_client/mqtt_configs.py`
## Workflow
In root directory:
```sh
docker compose up
```
- Run mosquitto broker

In `src/edge`:
```sh
python3 edge_initialization.py
```
- Run inference on model and save its inference times

In `src/edge`:
```sh
python3 run_edge.py
```
- Run edge

