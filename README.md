# Towards Closing the Loop in Robotic Pollination for Indoor Farming via Autonomous Microscopic Inspection

## Installation Guide
We utilize NITROS in NVIDIA Docker Environment. Please ensure that your computer has supported NVIDIA GPU.

### Windows 11 please start with the following pre-installation guide for WSL
First open a powershell and check nvidia driver status:
```
> nvidia-smi
```
TODO: finish this after experimenting with Rohan

### For Ubuntu 22.04 please starts here
1. Install Docker Engine from [this guild](https://docs.docker.com/engine/install/ubuntu/).

2. Configure Docker for rootless access [here](https://docs.docker.com/engine/install/linux-postinstall/).

3. Follow the [Developer Environment Setup](https://nvidia-isaac-ros.github.io/getting_started/dev_env_setup.html) to install nvidia-container-toolkit, Git LFS, and setup `~/workspaces/isaac_ros-dev/src` as `ISAAC_ROS_WS`.

For Step 1, select `On x86_64 platforms`.
For step 4, select `x86_64 and Jetson without SSD`.

### Install [ISAAC ROS Commons](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common)
```
cd ~/workspaces/isaac_ros-dev/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
```

### Obtain the repository
> **_NOTE:_**  make sure to clone it into the the right repo

We also want to rename the repo to ensure consistancy with ROS2 pkgs
```
cd ~/workspaces/isaac_ros-dev/src
git clone https://github.com/kczttm/IndoorFarming.git
mv ./IndoorFarming/ ./proj_farmhand/
```
Then navigate to the project root folder.
```
cd proj_farmhand/
```

With the Git LFS installed, we want to pull the large files 
```
git lfs pull
```
### Prepare to run the docker composition
Move the Isaac ROS Common Config file to home
```
cp ./.isaac_ros_common-config ~/
```
If you have a Kinova Gen3 arm, consider using our kinova controll library posted [here](https://github.com/kczttm/ros2_kinova_ws).
Otherwise please configure your docker environment following the guild from [here](https://nvidia-isaac-ros.github.io/concepts/docker_devenv/index.html#development-environment).

## Use Guide
### Entering Docker
We will add the following shortcut to build a docker env according to `~/.isaac_ros_common-config`
```
echo "alias ldb='cd ${ISAAC_ROS_WS}src/isaac_ros_common && ./scripts/run_dev.sh'" >> ~/.bashrc
echo "alias ld='cd ${ISAAC_ROS_WS}src/isaac_ros_common && ./scripts/run_dev.sh --skip_image_build'" >> ~/.bashrc
```
Note that `ldb` will build the docker first then launch it. `ld` will just launch what has already been built.

Now we start by with a fresh build:
```
source ~/.bashrc
ldb
```

### Inside Docker
For the first time building (or you have made changes non-python files under `${ISAAC_ROS_WS}src/`) build the package:
```
colcon build --symlink-install
source /workspaces/isaac_ros-dev/install/setup.bash
```

> **_NOTE:_** If you are using a desktop which has no built-in webcam, all OpenCV videocapture objects in this repo will need to be `device_id=n-2`. For example, `cv2.VideoCapture(2)` should be `cv2.VideoCapture(0)`.

Now to bring up all the hardware (realsense, endoscope, microscope):
we first want to give permission to the arduino (you could also have `/dev/ttyACM0` instead)
```
sudo chmod a+rw /dev/ttyUSB0
ros2 launch proj_farmhand multi_flower_pollination_demo.launch.py
```
Now you should see a running real time YOLO classifier on the endoscope camera.

To start the demo, please start a new terminal, attach to the running docker:
```
ld
```

then run the demo
```
ros2 run proj_farmhand main_multi_flower_pipeline
```