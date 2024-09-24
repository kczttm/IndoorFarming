# Towards Closing the Loop in Robotic Pollination for Indoor Farming via Autonomous Microscopic Inspection

## Installation Guide
We utilize NITROS in NVIDIA Docker Environment. Please ensure that your computer has supported NVIDIA GPU.

### Windows 11 please start with the following pre-installation guide for WSL
First open a powershell and check nvidia driver status:
```
> nvidia-smi
```

### For Ubuntu 22.04 please starts here
1. Install Docker Engine from [this guild]{https://docs.docker.com/engine/install/ubuntu/}.

2. Configure Docker for rootless access [here]{https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user}.

3. Follow the [Developer Environment Setup]{https://nvidia-isaac-ros.github.io/getting_started/dev_env_setup.html} to install nvidia-container-toolkit, Git LFS, and setup `~/workspaces/isaac_ros-dev/src` as `ISAAC_ROS_WS`.

### Obtain the repository
```
cd ~/workspaces/isaac_ros-dev/src
git clone https://github.com/kczttm/IndoorFarming.git
```
Then navigate to the project root folder.
```
cd IndoorFarming/
```

With the Git LFS installed, we want to pull the large files 
```
git lfs pull
```
### Prepare to run the docker composition
Move the Isaac ROS Common Config file to home
```
cp ./.isaac_ros_common-config ~
```
If you have a Kinova arm, consider using our kinova controll library posted [here]{https://github.com/kczttm/ros2_kinova_ws}. 
Otherwise please configure your docker environment following the guild from [here]{https://nvidia-isaac-ros.github.io/concepts/docker_devenv/index.html#development-environment}


