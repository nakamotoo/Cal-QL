# Cal-QL
This is the implementation for our paper [Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/abs/2303.05479) in Jax and Flax. 
- paper link: https://arxiv.org/abs/2303.05479
- project page: https://nakamotoo.github.io/projects/Cal-QL/
- video: https://youtu.be/r9CCdLeMJTg

This codebase is built upon [JaxCQL](https://github.com/young-geng/JaxCQL) repository.

If you find this repository useful for your research, please cite:

```
@article{nakamoto2023calql,
  author       = {Mitsuhiko Nakamoto and Yuexiang Zhai and Anikait Singh and Max Sobol Mark and Yi Ma and Chelsea Finn and Aviral Kumar and Sergey Levine},
  title        = {Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning},
  conference   = {arXiv Pre-print},
  year         = {2023},
  url          = {https://arxiv.org/abs/2303.05479},
}
```

## Installation
1. Install MuJoCo
- Download [MuJoCo key](https://www.roboti.us/license.html) and [MuJoCo 2.1 binaries](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)
- Extract the downloaded `mujoco210` and `mjkey.txt` into `~/.mujoco/mujoco210` and `~/.mujoco/mjkey.txt`

2. Add following environment variables into `~/.bashrc`
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

3. Install and use the included Ananconda environment
```
$ conda create -c nvidia -n Cal-QL python=3.8 cuda-nvcc=11.3
$ conda activate Cal-QL
$ pip install -r requirements.txt
```

4. Set up W&B API keys

This codebase visualizes the logs using [Weights and Biases](https://wandb.ai/site). To enable this, you first need to set up your W&B API key by: 
- Make a file named `wandb_config.py` under `JaxCQL` folder with the following information filled in
```
def get_wandb_config():
    return dict (
        WANDB_API_KEY = 'your api key',
        WANDB_EMAIL = 'your email',
        WANDB_USERNAME = 'user'
    )
```
You can simply copy [JaxCQL/wandb_config_example.py](JaxCQL/wandb_config_example.py), rename it to `wandb_config.py` and fill in the information.

## Run Experiments
### AntMaze
You can run experiments using the following command:
```
$ bash scripts/run_antmaze.sh
```
Please check [scripts/run_antmaze.sh](scripts/run_antmaze.sh) for the details.
All available command options can be seen in conservative\_sac_main.py and conservative_sac.py.

### Adroit Binary
1. Download the offline dataset from [here](https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y/view) and unzip the files into `<this repositroy>/demonstrations/offpolicy_hand_data/*.npy` 
2. We should also install `mj_envs` from [this fork](https://github.com/nakamotoo/mj_envs)
```
$ git clone --recursive https://github.com/nakamotoo/mj_envs.git
$ cd mj_envs  
$ git submodule update --remote
$ pip install -e .
```
3. Now you can run experiments using the following command:
 ```
$ bash scripts/run_adroit.sh
```
Please check [scripts/run_adroit.sh](scripts/run_adroit.sh) for the details.

### Other Environments
At the moment, this repository only has AntMaze and Adroit implemented. FrankaKitchen is planned to be added soon, but if you are in a hurry or would like to try other tasks (such as the visual manipulation domain in the paper), please contact me at nakamoto\[at\]berkeley\[dot\]edu.

## Sample Runs and Logs
In order to enable other readers to replicate our results easily, we have conducted a sweep for Cal-QL and CQL in the AntMaze and Adroit domains and made the corresponding W&B logs publicly accessible. The logs can be found here: https://wandb.ai/mitsuhiko/Cal-QL--Examples?workspace=user-mitsuhiko

You can choose the environment to visualize by filering on `env`. Cal-QL runs are indicated by `enable-calql=True`, and CQL runs are denoted by `enable-calql=False`. Each env has been run across 4 seeds.

## Credits
This project is built upon Young Geng's [JaxCQL](https://github.com/young-geng/JaxCQL) repository.
The CQL implementation is based on [CQL](https://github.com/aviralkumar2907/CQL).

In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 
