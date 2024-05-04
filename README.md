# IndoorFarming
Strawberry Indoor Pollination Project with Dr. Ai-Ping Hu GTRI

## Installation Guide
If no mamba, first install [miniforge](https://github.com/conda-forge/miniforge):
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/.bashrc
```

Otherwise proceed to create the environment.

Our video card is NVIDIA GeForce RTX 4090 with driver version v535.

Assuming no CUDA installed in BASE env, 
```
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then install the following python libraries:
```
mamba install opencv pandas scipy scikit-learn matplotlib
```
