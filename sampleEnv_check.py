"""
This tool is provided by stable_baselines3 to verify that your environment is working.
Run this file, and---if no errors are found---the environment is consistent and has a
correct syntax. The file will check your custom environment and output additional 
warnings if needed.

Common errors include having the observation space not agree with the output of an 
agent's step. This can be fixed by ensuring that such spaces are normalized and that all 
possible values of the outputs from the reset methods or step methods are within these.

Note: this file assumes that the custom environment you are checking is named 
"CustomEnv_Inverted".
"""
from stable_baselines3.common.env_checker import check_env
from CustomEnvironment_DPIC import DPICenv

env = DPICenv()
check_env(env)
