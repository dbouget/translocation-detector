# Python package for chromosome translocation detection in microscope images


## Installation

```
pip install git+https://github.com/dbouget/translocation-detector.git
pip install -e .
```


# Usage

## CLI
```
translocdet -c CONFIG -v debug
```

CONFIG should point to a configuration file (*.ini), specifying all runtime parameters,
according to the pattern from **blank_main_config.ini**.

## Python module
```
from translocdet import run_translocation_detection
run_translocation_detection(config_filename="/path/to/main_config.ini")
```

## Docker
When calling Docker images, the --user flag must be properly used in order for the folders and files created inside
the container to inherit the proper read/write permissions. The user ID is retrieved on-the-fly in the following
examples, but it can be given in a more hard-coded fashion if known by the user.

:warning: The following Docker image can only perform inference using the CPU. Another Docker image has to be created to
be able to leverage the GPU (see further down below). If the CUDA version does not match your machine, a new Docker image can be built manually, 
simply modifying the base torch image to pull from inside Dockerfile.

```
docker pull <user>/translocation-detector:v1.0-py38-cpu
```

For opening the Docker image and interacting with it, run:  
```
docker run --entrypoint /bin/bash -v /home/<username>/<resources_path>:/workspace/resources -t -i --network=host --ipc=host --user $(id -u) <user>/translocation-detector:v1.0-py38-cpu
```

The `/home/<username>/<resources_path>` before the column sign has to be changed to match a directory on your local 
machine containing the data to expose to the docker image. Namely, it must contain folder(s) with images you want to 
run inference on, as long as a folder with the trained models to use, and a destination folder where the results will 
be placed.

For launching the Docker image as a CLI, run:  
```
docker run -v /home/<username>/<resources_path>:/workspace/resources -t -i --network=host --ipc=host --user $(id -u) <user>/translocation-detector:v1.0-py38-cpu -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```

The `<path>/<to>/main_config.ini` must point to a valid configuration file on your machine, as a relative path to the `/home/<username>/<resources_path>` described above.
For example, if the file is located on my machine under `/home/myuser/Data/Translocation/main_config.ini`, 
and that `/home/myuser/Data` is the mounted resources partition mounted on the Docker image, the new relative path will be `Segmentation/main_config.ini`.  
The `<verbose>` level can be selected from [debug, info, warning, error].

For running models on the GPU inside the Docker image, run the following CLI, with the gpu_id properly filled in the configuration file:
```
docker run -v /home/<username>/<resources_path>:/workspace/resources -t -i --runtime=nvidia --network=host --ipc=host --user $(id -u) <user>/translocation-detector:v1.0-py38-cuda12.4 -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```


# Developers

For running inference on GPU, your machine must be properly configured (cf. [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html))  
In the configuration file, the gpu_id parameter should then point to the GPU that is to be used during inference.

To run the unit and integration tests, type the following within your virtual environment:
```
pip install pytest
pytest tests/
```
