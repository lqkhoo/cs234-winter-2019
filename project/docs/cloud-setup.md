This is mainly for my own reference to set up the dev env running on Ubuntu 18.04 LTS hosted on Amazon AWS. My workstation is a Windows 10 machine, so we will be using RDP (Remote Desktop) and putty.

Because of permission issues, especially when working with mujoco_py, we shall be working as root.

1. After launching the instance, download the priate key file (.pem)
2. Use putty's keygen to generate the .ppk file
3. In putty load the .ppk file and SSH to allocated IP address. Login as ubuntu
4. Make sure the following ports are open: 22(SSH), 3389(RDP), 6006(Tensorboard)


## Setup desktop and RDP
We shall use the lightweight xfce4 desktop environment.
1. SSH to instance as user ubuntu and run in order:
    ```
    sudo apt update
    sudo apt upgrade
    sudo apt install xrdp
    sudo apt install xfce4
    echo xfce4-session > .xsession
    sudo service xrdp restart
    ```
    * For AWS, do the following instead:
    ```
    sudo apt update
    sudo apt upgrade
    sudo apt install lxde
    sudo apt install xrdp
    ```
2. Now set the root password:
    ```
    sudo -i passwd
    ```
3. Open the RDP client and connect as root.
4. If your keyboard is anything other than English, and you want to work with a different layout on the instance, go to Applications->Settings->Keyboard->Layout and configure from there.
5. Grab a lightweight code editor
    ```
    sudo apt install gedit
    ```
6. Grab a browser
    ```
    sudo apt install firefox
    ```


## Setup CUDA

1. Setup directory structure
    ```
    cd ~
    mkdir dev && cd dev
    mkdir cs234 && cd cs234
    git clone https://github.com/lqkhoo/cs234-winter-2019.git .
    ```
2. Run the following setup scripts. This has been modified from the given script (for Azure). The first one gets CUDA and its dependencies and reboots the instance upon completion. The second scripts finishes configuring CUDA and tests running on GPU:
    ```
    #!/bin/bash

    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get -y update
    sudo apt-get -y install python3.6
    sudo apt-get -y install python-pip
    sudo pip install --upgrade virtualenv
    sudo apt-get -y install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev curl
    sudo apt-get -y install ffmpeg
    curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    sudo apt-get -y update
    sudo apt-get -y install cuda-9-0

    echo >> ~/.bashrc '
    alias python=python3.6
    '
    source ~/.bashrc
    sudo reboot
    ```

    If there are errors, refer to following block. These errors occured when setting up as non-root user:
    ```
    sudo apt install --fix-broken
    
    # Force overwrite to resolve conflict between cuda9.0 and libglx-mesa
    # https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-390/+bug/1753796
    sudo apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken

    # cuda 9.0 needs gcc <= version 6, so install that
    # https://github.com/ethereum-mining/ethminer/issues/731
    sudo apt-get -y install g++-6
    sudo apt-get -y install gcc-6
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
    ```



    And then

    ```
    #!/bin/bash

    curl -O https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
    curl -O https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb

    sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
    sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
    sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb

    echo >> ~/.bashrc '
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda9.0/lib64:/usr/local/cuda/extras/CUPTI/lib64"
    export CUDA_HOME=/usr/local/cuda9.0
    export PATH="$PATH:/usr/local/cuda9.0/bin"
    '
    source ~/.bashrc

    sudo apt-get -y install cmake
    sudo apt-get -y install zlib1g-dev
    python3.6 -m pip install --upgrade -r requirements.txt

    cp -r /usr/src/cudnn_samples_v7/ .
    cd cudnn_samples_v7/mnistCUDNN
    make clean && make
    ./mnistCUDNN
    ```

## Setup Python env
1. Setup virtualenv
    ```
    sudo apt install python3-virtualenv
    cd <projectdir>
    python3 -m venv <venvname>
    source <venvname>/bin/activate
    ```

## Setup mujoco

1. Setup mujoco
    ```
    cd ~
    mkdir .mujoco
    ```

    Get mujpro150linux from https://www.roboti.us/index.html
    ```
    sudo apt unzip
    cd ~/Downloads
    unzip mjpro150_linux.zip -d ~/.mujoco
    ```
    Get the mujoco license key, then
    ```
    cp ~/Downloads/mjkey.txt ~/.mujoco/mjpro150/bin # This is for running mujoco on its own
    cp ~/Downloads/mjkey.txt ~/.mujoco/ # This is for mujoco_py
    ```
    Test the installation
    ```
    cd ~/.mujoco/mjpro150/bin
    ./simulate ~/.mujoco/mjpro150/model/humanoid.xml
    ```

## Setup mujoco_py
1. Installing directly with ```pip install mujoco_py``` seems to have problems when doing ```import mujoco_py```, so let's build it from source.

    Download the zip from https://github.com/openai/mujoco-py/
    ```
    cd ~/Downloads
    unzip ~/mujoco-py-master.zip .
    ```
    Make sure you're in the right virtualenv.
    ```
    source <venv>/bin/activate
    pip install wheel
    cd ~/Downloads/mujoco-py-master
    pip install -r requirements.dev.txt
    pip install -r requirements.txt
    sudo python3 setup.py install
    sudo apt install libglew-dev
    sudo apt install patchelf
    sudo apt install libglfw3-dev
    ```

    ```
    gedit ~/.bashrc
    ```
    Append to the file
    ```
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia-410"
    ```

    Now open a fresh terminal and try all of the following in the python interpreter:
    ```
    import numpy as np
    import tensorflow as tf
    import gym
    import mujoco_py
    ```