# Reinforcement: Inverting and Relaxing a Double-Pendulum In a Cart (DPIC)
Repository created by Adan J. Mireles (Fall 2022) for the Data-Enabled Physics Course at Rice University.

- Derive equations of motion for a Double-Pendulum In a Cart (DPIC).

- Define initial conditions and animate a DPIC by solving a non-linear system of ODEs using `sympy`'s function `odeint`.

- Create custom environments for training agent to hold pendulum at inverted position or relax it by holding it down

- Train an agent (via reinforcement learning, powered by [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html)) to invert or relax a double-pendulum by controlling the horizontal force imposed on the cart. 

## Instructions:
- Make sure you have fulfilled all requirements in the `requirements.txt` file
- Download files into a single folder
- [Optional] Verify that CUDA device is enabled by running the corresponding `verifyCUDA.py` file (see "CUDA devices section below")
- [Optional] Verify that custom environment does not show any warnings (run `sampleEnv_check.py`). If nothing appears after running, that is good.
- Open main executable ".py" file (`train_model_DPIC.py`). The file is set as default to train an agent for 200k episodes and save 20 models linearly spaced (that is, to save one model every 10k episodes and a visualization therein). These can be easily changed by changing the "DEFAULT PARAMETERS" in the first few lines after the description docstring. Note that the algorithm can be changed at will. See [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html) for possible algorithms, note that the algorithm must be such that continuous action/observation spaces are possible.  
- Run `train_model_DPIC.py`. Note that a unique timestamp is assigned to the run, which is used to identify the agent being trained. 
- [Optional] Open **TensorBoard** to see progress of run (on command window, type: `tensorboard --logdir ./{path-to-logs-directory}`)—**this part generates plots of agent's reward/episode length versus episode number**
- After run is finished, go to the "models" folder created by the run, within which a folder named with a timestamp will be present (and all other timestamps, if file is ran more than once). There, .zip files with unique timestamps will be present (can be used to load a model for further training or testing), as well as .gif files with explicit names (e.g., "DDPG_110000", which is an animation of the model being trained after 110000 episodes under the DDPG algorithm).

## CUDA Devices:
In order to boost agent training...
- Check if your system has CUDA available (you can verify on [NVIDIA CUDA-Zone](https://developer.nvidia.com/cuda-zone))
- If available, download the most recent CUDA toolkit from [here](https://developer.nvidia.com/cuda-downloads)
