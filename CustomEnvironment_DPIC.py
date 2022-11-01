#Packages required by DPIC simulation
import numpy as np
import sympy as smp
from sympy.solvers.solveset import linsolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

# RL packages
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# Importing DPIC equations of motion
from DPIC_vars_model import (dz0dt_f, dz1dt_f, dz2dt_f, 
                             dthe0dt_f, dthe1dt_f, dthe2dt_f)

# Other packages
import cv2

# In[]: Environment Parameters

# Ultimate goal is to hold pendulum at inverted position for some time:
hold_time_goal = 0.8    #sec
max_stretch_tol = 0.98  #98% of pendulum's maximum (vertical) stretch  
                        #qualifies as an inverted pendulum

###################
# Time Parameters #
###################

# Max time to fail (timeout)
t_out = 10 # sec

# Duration of constant action:
t_action = 0.05 # sec

# Number of calculations 
num_calcs = 6 

# Times for which the solver will compute the state of the DPIC
t = np.linspace(0, t_action, num_calcs)  # num_calcs/t_action = 80 calculations per second


# In[]: 

# Within this definition, "S" is the vector containing the state of the DPIC
# "dSdt" returns the rate of change of each component of the S vector

def dSdt(S, t, u, g, m0, m1, m2, L1, L2):
    the0, z0, the1, z1, the2, z2 = S
    return [
        dthe0dt_f(z0),
        dz0dt_f(t, u, g, m0, m1, m2, L1, L2, the0, the1, the2, z0, z1, z2),
        dthe1dt_f(z1),
        dz1dt_f(t, u, g, m0, m1, m2, L1, L2, the0, the1, the2, z0, z1, z2),
        dthe2dt_f(z2),
        dz2dt_f(t, u, g, m0, m1, m2, L1, L2, the0, the1, the2, z0, z1, z2),
    ]

def odeSol(S,u):
    the0, z0, the1, z1, the2, z2 = S
    sol = odeint(dSdt, y0=S, t=t, args=(u, g, m0, m1, m2, L1, L2))
    return sol

# u=2
# solAA = odeSol([0,0,np.pi/2,0,np.pi/2,0])
# solAA[5]

# In[]: 

# Defining what would cause DPIC to succeed.
# DPIC is rewarded if the second pendulum is held in inverted position 
# for a defined time.

# Here, theta1 and theta2 are angles of DPs at current state
def invert_check(time_held,theta1,theta2):
       # y-coordinate of second end of DPIC    >= some fraction of max stretch 
    if (L1*np.cos(theta1) + L2*np.cos(theta2)) >= max_stretch_tol*maxL:
        time_held += t_action
        if time_held >= hold_time_goal:
            return 2   # Agent successfully inverts DPIC for goal time
                       # if invert_check == 2: add t_action to time_held and give high reward
        else:
            return 1   # Agent inverts DPIC but not for goal time yet
                       # if invert_check == 1: add t_action to time_held and give reward
    else:
        return 0       # Agent has not inverted DPIC
                       # if invert_check == 0: time_held = 0 and punish

# Defining what would cause DPIC to fail (x-axis limits):
# Here, theta0 is position of cart on current state
def out_of_bounds(theta0):
    if theta0 >= max_bd or theta0 <= min_bd:
        return 1
    else: 
        return 0

# Defining what would cause DPIC to fail (time limit):
def timeout(time_elapsed):
    if time_elapsed >= t_out:      # Defined within Environment Parameters
        return 1
    else:
        return 0
    
# In[]: Initial Conditions

# Physical Parameters
g = 9.81        #m/sec^2   Gravitational acceleration
m0 = 2.1        #kg        Mass of cart
m1= 0.7         #kg        Mass of pendulum 1
m2= 0.7         #kg        Mass of pendulum 2
L1 = 0.8        #m         Length of rod 1
L2 = 0.8        #m         Length of rod 2
maxL = L1 + L2  #m         Maximum stretch

# Control Force Limits
max_u = m0*g    #N     Maximum control force
min_u = -max_u  #N     Minimum control force

# Initial angle displacement
theta_small = np.pi/50

# Bound Limits
max_bd = 1.5*maxL    #m
min_bd = -max_bd   #m

# Angle limits
the1_lim = 10*np.pi
the2_lim = 25*np.pi

# In[]: Defining the Environment

class DPICenv(gym.Env):

    def __init__(self):
        super(DPICenv, self).__init__()
        
        # The action space is only to include a non-discrete value in
        # a normalized range [-1, 1]. Actual range is [min_u,max_u]...
        
        self.u_high = 2*max_u
                
        self.action_space = spaces.Box(low=-self.u_high, high=self.u_high, 
                                       shape=(1,), dtype=np.float32)
        
        # Observations to include (total 6):
        # (*) Position of cart (1 value)
        # (*) Speed of cart (1 value)
        # (*) Position of each pendulum (2 values)
        # (*) Angular speed of each pendulum (2 values)
        
        self.x_bound = 2*max_bd
        self.the1_bd = 2*the1_lim
        self.the2_bd = 2*the2_lim
        
        high = np.array(
            [self.x_bound,               # Horizontal movement bounds
             np.finfo(np.float32).max,   # Value simulating infinity
             self.the1_bd,               # 
             np.finfo(np.float32).max,   # Value simulating infinity
             self.the2_bd,               # 
             np.finfo(np.float32).max,   # Value simulating infinity
            ],
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(-high, high, 
                                            dtype=np.float32)

    def step(self, action): 
        assert self.action_space.contains(
        action), f"{action!r} ({type(action)}) invalid"
		
        # self.prev_actions.append(action)
        
        self.u = action*max_u  # Undo normalization
        
        # Calculate ODE solution 
        self.sol = odeSol( [self.cart_pos, self.cart_speed, 
                            self.pen1_ang, self.pen1_angVel, 
                            self.pen2_ang, self.pen2_angVel],self.u)
        self.S = self.sol[num_calcs - 1]
        self.time_elapsed = self.time_elapsed + t_action
        
        ########
        # Fail #
        ########
        
        # What to do if out of bounds
        
        if out_of_bounds(self.cart_pos) == 1 or timeout(self.time_elapsed) == 1:
            self.done = True
            
        ycoords = L1*np.cos(self.pen2_ang) + L2*np.cos(self.pen2_ang)
        sum_vel = self.cart_speed + self.pen1_angVel + self.pen2_angVel
                
        self.total_reward = ycoords - sum_vel
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward
        
        if self.done:
            self.reward = -10
        info = {}
        
        cartPos = self.cart_pos/(2*max_bd)
        cartSpeed = self.cart_speed/(4*max_bd)
        pen1Ang = self.pen1_ang/(10*np.pi)
        pen1AngVel = self.pen1_angVel/(200*np.pi)
        pen2Ang = self.pen2_ang/(25*np.pi)
        pen2AngVel = self.pen2_angVel/(500*np.pi)
        
        observation = [cartPos, cartSpeed, 
                       pen1Ang, pen1AngVel, 
                       pen2Ang, pen2AngVel]
        observation = np.array(observation)
        
        return observation, self.reward, self.done, info
    
    
    def reset(self):
        # Initializing DPIC conditions
        self.u = 0
        self.time_elapsed = 0
        
        # Initial state of DPIC
        self.cart_pos = 0.
        self.cart_speed = 0.
        self.pen1_ang = 0. + theta_small
        self.pen1_angVel = 0.
        self.pen2_ang = 0. + theta_small
        self.pen2_angVel = 0.
        
        # Initial state in single list
        self.S = [self.cart_pos, self.cart_speed, 
                  self.pen1_ang, self.pen1_angVel, 
                  self.pen2_ang, self.pen2_angVel]
        
        self.prev_reward = 0
        
        self.done = False
             
        # Create observation:
        
        # cartPos = self.cart_pos/(2*max_bd)
        # cartSpeed = self.cart_speed/(4*max_bd)
        # pen1Ang = self.pen1_ang/(10*np.pi)
        # pen1AngVel = self.pen1_angVel/(200*np.pi)
        # pen2Ang = self.pen2_ang/(25*np.pi)
        # pen2AngVel = self.pen2_angVel/(500*np.pi)
        
        # cartPos = self.cart_pos
        # cartSpeed = self.cart_speed
        # pen1Ang = self.pen1_ang
        # pen1AngVel = self.pen1_angVel
        # pen2Ang = self.pen2_ang
        # pen2AngVel = self.pen2_angVel
        
        observation = np.array(self.S,dtype=np.float32)
        return observation  # reward, done, info can't be included
