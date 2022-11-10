# Tool used to ensure that the custom environment is working
# It will check your custom environment and output additional warnings if needed
# This assumer that the custom environment is named "CustomEnvironment_DPIC"

from stable_baselines3.common.env_checker import check_env
from CustomEnvironment_DPIC import DPICenv

env = DPICenv()
check_env(env)