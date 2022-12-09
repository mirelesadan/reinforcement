# Reinforcement: Inverting and Relaxing a Double-Pendulum In a Cart (DPIC)
Repository created by Adan J. Mireles (Fall 2022) for the Data-Enabled Physics Course at Rice University.

- Derive equations of motion for a Double-Pendulum In a Cart (DPIC).

- Define initial conditions and animate a DPIC by solving a non-linear system of ODEs using sympy's function odeint.

- Create custom environments for training agent to hold pendulum at inverted position or relax it by holding it down

- Train an agent (via reinforcement learning) to invert or relax a double-pendulum by controlling the horizontal force imposed on the cart. 

## CUDA Devices:
In order to boost agent training...
- Check if your system has CUDA available (you can verify on https://developer.nvidia.com/cuda-zone)
- If available, download the most recent CUDA toolkit from https://developer.nvidia.com/cuda-downloads
