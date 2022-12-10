"""
[Main, executable .py file.] 
Upon running:
    (1) Directories for models and logs will be created in specified directory.
    (2) Reinforcement learning algorithm is loaded*
    (3) Model learning progress is logged in tensorboard (type the follwing
        command on terminal: "tensorboard --logdir .\{LogDirectory}" to see)
    (4) Agent is trained** for user-defined number of episodes (default is 200k) 
        and predefined number of models are automatically saved during training
        (default is 20, thus saving a model every 10k episodes)
    (5) A visualization (.gif file) is saved every time a model is saved
    
* If different reinforcement learning algorithm is desired, replace all "DDPG"
  entries by deired algorithm (e.g., "PPO" and "A2C").
** Pre-trained model can further be trained. See lines 98-100.
"""
# Default parameters
tot_episodes = 200000         # Total number of episodes to train model for
num_saves = 20                # Number of models and animations saved
FPS = 25                      # Frames per second in animations saved

# Specify the algorithm and number of timesteps before saving trained agent
# Load custom environment and train agent using specified algorithm

from stable_baselines3 import DDPG    # Choose to import different algorithm if desired (e.g., PPO instead of DDPG)
import os
from CustomEnv_Relaxed import DPICenv, t_out, t_action, L1, L2
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.animation import PillowWriter

# Quick work-around for OMP error Initializing libiomp5md.dll.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def xy_coords(t, theta0, theta1, theta2, Len1, Len2):
    """
    Input the position of the cart, the angle of both pendulums with respect 
    to the vertical, and the length of each rod.
    
    Parameters
    ----------
    t : float
        Time at which the positions want to be known..
    theta0 : float
        x-position of the cart.
    theta1 : float
        Angle of pendulum 1 with respect to the vertical.
    theta2 : float
        Angle of pendulum 2 with respect to the vertical.
    Len1 : float
        Length of rod from first pendulum.
    Len2 : float
        Length of rod from second pendulum.

    Returns
    -------
    The (x,y) coordinates of the cart and the end of both pendulums as a 6-tuple.

    """
    return (theta0,
            0*theta0,
            theta0 + Len1*np.sin(theta1),
            Len1*np.cos(theta1),
            theta0 + Len1*np.sin(theta1) + Len2*np.sin(theta2),
            Len1*np.cos(theta1) + Len2*np.cos(theta2))

# Some constants for animation
Lsum = L1 + L2
Lmax = Lsum + 0.1
Lmin = -Lmax
cartX = 0.2
cartY = 0.1
wheelR = 0.03

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
time_tag = f"{int(time.time())}"


# Create directories in which models are to be saved and in which logs 

if not os.path.exists(models_dir):
 	os.makedirs(models_dir)

if not os.path.exists(logdir):
 	os.makedirs(logdir)

env = DPICenv()
env.reset()

# Change "DDPG" below to use different algorithm (if desired)
model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# If training pre-trained model, uncomment lines below and comment the one above:
# model = DDPG.load("./path-of-model", env=env, verbose=1, tensorboard_log=logdir)
# env = model.get_env()

num_gifs = num_saves       # Number of models and visualizations saved
TIMESTEPS = int(tot_episodes/num_gifs)

print(f"Agent will learn for a total of {tot_episodes} episodes.")
print(f"One model and animation will be saved for every {int(tot_episodes/num_gifs)} episodes.")
print(f"A total of {num_gifs} models and animations will be saved on the directory: ./{models_dir}")

# TRAIN AND SAVE MODELS
for i in range(num_gifs):
	iters = i + 1                                              
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
	print(f"Agent has learned for {TIMESTEPS*iters} episodes.")
    
    # LOAD, TEST, AND VISUALIZE MODEL PERFORMANCE AFTER "TIMESTEPS*iters" EPISODES
	env2 = DPICenv()
	iters = i + 1
	model2 = DDPG.load(f"./models/{time_tag}/{TIMESTEPS*iters}", env=env2)
	history = np.empty((0,6),np.float32)  # Initialize history of states
	obs = env2.reset()                     # Initial condition
        
	for j in range(int(t_out/t_action)):
		prev_state = obs
		history = np.vstack((history, prev_state))
		action, _states = model2.predict(obs, deterministic=True)
		obs, rewards, done, info = env2.step(action)
		if done:
			obs = env2.reset()
    
	t = np.linspace(0, t_out, len(history))
	the0 = history.T[0]*(Lsum)
	the1 = history.T[2]*np.pi    # Undo normalization
	the2 = history.T[4]*np.pi    # Undo normalization
    
    # Plot and save figures
	plt.plot(t, the0, label='$\\theta_0$, Cart Position')
	plt.plot(t, the1, label='$\\theta_1$, Pendulum #1')
	plt.plot(t, the2, label='$\\theta_2$, Pendulum #2')
	plt.legend()
	plt.xlabel('Time (sec)')
	plt.ylabel('$\\theta_i$ (rad) for $i>0$, $(m)$ if $i=0$')
	plt.savefig(f'./models/{time_tag}/DDPG_{TIMESTEPS*iters}.svg')
	plt.savefig(f'./models/{time_tag}/DDPG_{TIMESTEPS*iters}.pdf')
	plt.clf()
    
    # Define variables for animation
	the0_min = min(the0)-Lsum
	the0_max = max(the0)+Lsum
	x0, y0, x1, y1, x2, y2 = xy_coords(t, the0, the1, the2, L1, L2)
	xy_corners = np.stack((x0 - cartX/2, y0 - cartY/2), axis=-1)                  # xy coordinates of bottom-left corner of cart
	leftW_centers = np.stack((x0 - cartX/4, y0 - (cartY/2 + wheelR)), axis=-1)    # xy coordinates of left wheel of cart
	rightW_centers = np.stack((x0 + cartX/4, y0 - (cartY/2 + wheelR)), axis=-1)   # xy coordinates of right wheel of cart
    
	patch = Rectangle(xy_corners[0], cartX, cartY,    # Cart body drawing
                        edgecolor = 'k',
                        facecolor = 'silver',
                        fill=True,
                        lw=1)
	patchW1 = Circle(leftW_centers[0], wheelR,        # Left wheel drawing
                        edgecolor = 'k',
                        facecolor = 'dimgrey',
                        fill=True,
                        lw=1)
	patchW2 = Circle(rightW_centers[0], wheelR,       # Right wheel drawing
                        edgecolor = 'k',
                        facecolor = 'dimgrey',
                        fill=True,
                        lw=1)
    
	def init():
		return(patch, patchW1, patchW2),
    
	def animate(k):
		    ln1.set_data([x0[k], x1[k], x2[k]], [y0[k], y1[k], y2[k]])
		    patch.set_xy(xy_corners[k])
		    patchW1.set_center(leftW_centers[k])
		    patchW2.set_center(rightW_centers[k])
		    ax.add_patch(patch)
		    ax.add_patch(patchW1)
		    ax.add_patch(patchW2)
                
	fig, ax = plt.subplots(1,1, figsize=(the0_max-the0_min,2*Lmax))
            
	ln1, = plt.plot([], [], 'ko-', lw=1, markersize=2)
	ax.set_ylim(Lmin,Lmax)
	ax.set_xlim(the0_min,the0_max)
    
	patches = [patch, patchW1, patchW2] 
	for p in patches:
		ax.add_patch(p)
            
        # Cart's "track"
	ax.plot([the0_min,the0_max],[-(cartY/2+2*wheelR),-(cartY/2+2*wheelR)],'k-',lw=1)  
    
	ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,len(history),int(len(history)/FPS/t_out)), interval=10)
	ani.save(f"./models/{time_tag}/DDPG_{TIMESTEPS*iters}.gif", writer='pillow', fps=FPS)
        # ani.save("./models/1669973706/DDPG_180000_{iters}.gif", writer='pillow', fps=FPS)
	plt.clf()

	print(f"The performance of agent trained after {TIMESTEPS*iters} episodes can now be visualized.")
	print(f"Go to directory: ./models/{time_tag}/DDPG_{TIMESTEPS*iters}.gif")
	env2.close()
    

env.close()
