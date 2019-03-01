This configures our environment for the project, which is a Windows 10 workstation with 64GB memory, Intel i7-8700K (3.70GHz, 6 cores, 12 Hyperthreads), running a single nVidia GTX1080 GPU. This should be roughly equivalent to a single high large GPU instance on the cloud. For storage, we have a Samsung NVMe 960 EVO with QD1 R/W=14k/50kIOPS to QD32 R/W=380k/360kIOPS, so we could reasonably use it as a cache.

We shall be using Anaconda and Pytorch, alongside mujoco_py, which we will build from source. We shall be using CUDA 10.0

1. Install CUDA 10 from https://developer.nvidia.com/cuda-downloads
1. Install mujoco under ```%USERNAME%/.mujoco```. Put license key file mjkey.txt under ```/.mujoco```, and inside ```/bin``` of each installation.
1. Install Anaconda from https://www.anaconda.com/distribution/#download-section. Note the installation path
1. Add path to conda and /Scripts to environment variables.
1. ```conda upgrade --all```
1. Clone/download mujoco_py from https://github.com/openai/mujoco-py
1. Execute within mujoco-py-master:
    ```
    pip install -r requirements.py
    pip install -r requirements.dev.py
    python setup.py install
    ```
1. Now install pytorch:
    ```
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    ```
1. Ensure imports within the interpreter:
    ```
    import numpy as np
    import torch
    import mujoco_py
    ```