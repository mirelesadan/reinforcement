from CustomEnvironment_DPIC import DPICenv
from stable_baselines3 import PPO
import numpy as np

env = DPICenv()
model = PPO.load("./models/1667978871/100000", env=env)

history = np.empty((0,6),np.float32)
obs = env.reset()

j = 0
for i in range(333):
    prev_state = obs
    history = np.vstack((history, prev_state))
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    if done:
        j = j + 1
        obs = env.reset()

print(str(history.tolist()))
print("Num. fails =",j)