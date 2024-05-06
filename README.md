# IndoorFarming
Strawberry Indoor Pollination Project with Dr. Ai-Ping Hu GTRI

## Installation Guide
### If no [uv](https://github.com/astral-sh/uv)

Install uv with curl
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env
```

### Otherwise proceed to clone the repository.
```
git clone https://github.com/kczttm/IndoorFarming.git
```
Then navigate to the project root folder.

### Obtain [git-lfs](https://packagecloud.io/github/git-lfs/install)
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```
Then pull the large files 
```
git lfs pull
```

Create a virtual environment within the project folder and enter
```
uv venv
source ./.venv/bin/activate
```

Our video card is NVIDIA GeForce RTX 4090 with driver version v535.
Assuming no CUDA installed in BASE env, 
```
uv pip install torch torchvision torchaudio
```

Then install the following python libraries:
```
uv pip install opencv-python pandas scipy scikit-learn matplotlib tqdm ipykernel ultralytics plotly
```

