This configures our environment for the project, which is a Windows 10 Pro workstation with 64GB memory, Intel i7-8700K (3.70GHz, 6 cores, 12 Hyperthreads), running a single nVidia GTX1080 GPU (3.7 compute units). This should be equivalent to a single large GPU instance on the cloud.

For storage, we have a Samsung NVMe 960 EVO with QD1 R/W=14k/50kIOPS to QD32 R/W=380k/360kIOPS.

We shall be using Anaconda and Pytorch, alongside mujoco_py, which we will build from source. We shall be using CUDA 10.0

1. Install CUDA 10 from https://developer.nvidia.com/cuda-downloads
1. Install mujoco under ```%USERNAME%/.mujoco```. Put license key file mjkey.txt under ```/.mujoco```, and inside ```/bin``` of each installation. We will be working with mujoco1.5.
1. Install Anaconda from https://www.anaconda.com/distribution/#download-section. Note the installation path
    * Add path to conda and /Scripts to environment variables.
    * ```conda upgrade --all```
    * Before using python or pip in any form, basically python commands other than ```conda``` itself, make sure to do ```conda activate``` to use the base env, otherwise modules may fail to load.
1. Install mujoco_py.
    * Clone/download mujoco_py from https://github.com/openai/mujoco-py
    * Follow https://github.com/openai/mujoco-py/issues/253#issuecomment-446025520. Install Microsoft Visual Studio C++ Build Tools 2015.
    * Use the Visual C++ 2015 x64 Native Build Tools Command Prompt from ```C:\Program Files (x86)\Microsoft Visual C++ Build Tools```. cd to ```mujoco-py-master``` and execute:
        ```
        pip install -r requirements.txt
        pip install -r requirements.dev.txt
        python setup.py install
        ```
    * If there are errors try: https://github.com/openai/mujoco-py/issues/253#issuecomment-446025520
1. Install deepmind control suite for mujoco150. Generally dm_control and mujoco_py are separate and incompatible wrappers to mujoco, so you will have to pick between one or the other. We will be borrowing the models (xml files) to be included in our assets in either case, however. Not sure whether this needs to be run from within the native build tools command prompt but there's no harm in doing so.
    * Get the source from https://github.com/deepmind/dm_control/releases/tag/mujoco1.50
        ```
        pip install -r requirements.txt
        python setup.py install
        ```
    * Now set the environment variable 'MUJOCO_GL' to the value 'glfw'. Refer to:
        * https://github.com/deepmind/dm_control/issues/78
        * https://github.com/deepmind/dm_control/commit/276c6e8ce6dbcb8a037f8aa6a76475db4da41e6a
1. Install pytorch.
    ```
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    ```
1. Ensure imports within the interpreter:
    ```
    import numpy as np
    import torch
    import mujoco_py
    import dm_control
    ```
1. Run one of the samples from mujoco_py and make sure it works.
1. Additional dependencies:
    * ```pip install imageio-ffmpeg``` to record video. Write still fails silently for mujoco_py. We'll use some other means like OBS to record.