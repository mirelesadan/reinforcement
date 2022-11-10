# Reinforcement: Inverting a Double-Pendulum In a Cart (DPIC)
Repository created by Adan J. Mireles (Fall 2022) for the Data-Enabled Physics Course at Rice University.


- Derive equations of motion for a Double-Pendulum In a Cart (DPIC).

- Define initial conditions and animate a DPIC by solving a non-linear system of ODEs using sympy's function odeint.

- Create a custom environment for training agent to hold pendulum at inverted position

- Train an agent (via reinforcement learning) to invert a double-pendulum by controlling the horizontal force imposed on the cart. 

## Requirements:

- Install pytorch (https://pytorch.org/get-started/locally/) 

- Install stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
  - Using pip:  ``` pip install stable-baselines3[extra] ```: the "```[extra]```" option automatically adds TensorBoard, ```pyAtari```, and OpenCV.

- It is extremely recommended download TensorBoard as well (https://pypi.org/project/tensorboard/), as it enables to track and visualize agent learning metrics.  
  - Using pip:  ``` pip install tensorboard ```
  - When training an agent, run the following in a command line: ```tensorboard --logdir ./[name_of_directory_with_log_folder]/```. 

## Other:

- CUDA devices
  - Check if your system has such available on https://developer.nvidia.com/cuda-zone
  - If available, download the most recent CUDA toolkit from https://developer.nvidia.com/cuda-downloads
