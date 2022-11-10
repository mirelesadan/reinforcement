# =============================================================================
# Double Pendulum in Cart (DPIC)
# ~ Custom Environment ~
# by Adan J. Mireles Fall 2022
# =============================================================================

# IMPORT NECESSARY PACKAGES

# Packages required by DPIC simulation/data generation
import numpy as np
from scipy.integrate import odeint

# RL packages
import gym
from gym import spaces


# DPIC equations of motion
from DPIC_vars_model import (dz0dt_f, dz1dt_f, dz2dt_f, 
                             dthe0dt_f, dthe1dt_f, dthe2dt_f)

# DEFAULT PARAMETERS:
# Physical parameters
g = 9.81          # m/sec^2  Gravitational acceleration
m0 = 2.1          # kg       Mass of cart
m1= 0.7           # kg       Mass of pendulum 1
m2= 0.7           # kg       Mass of pendulum 2
L1 = 0.8          # m        Length of rod 1
L2 = 0.8          # m        Length of rod 2
# Action Parameters
t_action = 0.006   # sec     Duration of constant action
num_calcs = 4      # per timestep
t = np.linspace(0, t_action, num_calcs)    # Previosuly 0, 0.03 , 6
# Success/Fail Parameters
t_out = 10       # sec   Timeout
hold  = 0.6      # sec   Time after which agent starts getting a higher reward if inverted
stretch = 0.995  # % of pendulum's vertical stretch 


# HELPER FUNCTIONS:
# Return the rate of change of each component of the S vector
# Within this definition, "S" is the vector containing the state of the DPIC
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

# Compute state of DPIC at times specified in "t" (See Action Parameters above)
def odeSol(S,u):
    the0, z0, the1, z1, the2, z2 = S
    sol = odeint(dSdt, y0=S, t=t, args=(u, g, m0, m1, m2, L1, L2))
    return sol

# Extract position of each component on system
# Input: (time, cartPos, AnglePen1, AnglePen1, LengthPen1, LengthPen2)
def xy_coords(time, the0, the1, the2, L1, L2):
    return (the0,                                      # Cart x-position
            0*the0,                                    # Cart y-position
            the0 + L1*np.sin(the1),                    # Pendulum 1 x-position
            L1*np.cos(the1),                           # Pendulum 1 y-position
            the0 + L1*np.sin(the1) + L2*np.sin(the2),  # Pendulum 2 x-position
            L1*np.cos(the1) + L2*np.cos(the2))         # Pendulum 2 y-position

# Normalize pendulum angle x between [-pi, pi]
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

# Agent Failre/Success Conditions:
# Condition (1): DP inversion
def invert_check(time_held, hold_time_goal, pen1_angle, pen2_angle):
    ycoord_pen2 = L1*np.cos(pen1_angle) + L2*np.cos(pen2_angle)          
    if ycoord_pen2 >= (L1 + L2) * stretch:
        if time_held >= hold_time_goal: # Agent successfully inverts DPIC for goal time
            return 2                    # if invert_check == 2: add t_action to time_held       
        else:                           # Agent inverts DPIC but not for goal time yet
            return 1                    # if invert_check == 1: add t_action to time_held
    else:                               # Agent has not inverted DPIC
        return 0                        # if invert_check == 0: time_held = 0
     
# Condition (2): Determine whether cart is within or out of bounds
def out_of_bounds(cart_pos, bound):                                                                         
    if cart_pos >= bound or cart_pos <= -bound:
        return 1  # Agent is out of bounds
    else: 
        return 0  # Agent is within bounds

# Condition (3): Determine whether cart is moving too fast
def speed_Lim(cart_speed, limit):
    if cart_speed >= limit:
        return 1  # Cart is moving too fast
    else: 
        return 0  # Cart speed is okay    

# Condition (4): Determine whether cart is moving too fast
def angVel_Lim(angVel1, angVel2, limit):
    if angVel1 >= limit or angVel2 >= limit:
        return 1  # Cart is moving too fast
    else: 
        return 0  # Swing rate is a 

# Condition (5): Verify if agent has exhausted its time
def timeout(time_elapsed):
    if time_elapsed >= t_out:
        return 1  # Agent unable to invert pendulum within allowed time
    else:
        return 0  # Agent still trying to invert pendulum within allowed time

class DPICenv(gym.Env):

    def __init__(self):     
        # Some Physical Parameters, defined 
        self.g = g                     # m/sec^2  Gravitational acceleration
        self.m0 = m0                   # kg       Mass of cart
        self.m1= m1                    # kg       Mass of pendulum 1
        self.m2= m2                    # kg       Mass of pendulum 2
        self.L1 = L1                   # m        Length of rod 1
        self.L2 = L2                   # m        Length of rod 2
        self.maxL = self.L1 + self.L2  # m        Maximum DP stretch
        
        # Time Parameters 
        self.t_out = t_out              # sec         Max time to fail (timeout)
        self.t_action = t_action        # sec         Duration of constant action
        self.num_calcs = num_calcs      # per timestep
        self.hold_time_goal = hold      # sec         Time for which agent must hold pendulum at inverted position)
        self.max_stretch_tol = stretch  # 99.9% of pendulum's vertical stretch 

        # Times for which the solver will compute the state of the DPIC
        self.t = t  
        
        # DIPC System Bounds/Limits/Restrictions
        self.u_high = 25 * self.m0 * self.g   # N        Maximum control force
        self.xmax_bd = self.maxL              # m        Right bound of horizontal movement
        self.spd_bd = 5 * self.xmax_bd        # m/sec    Maximum cart speed
        self.the1_bd = np.pi                  # rad      Bounds for angles on [-pi, pi]
        self.the2_bd = np.pi                  # rad      Bounds for angles on [-pi, pi]
        self.angVel_bd = (2*np.pi) * 10       # rad/sec  Also (2*pi) * N rev/s
        
        # ACTION SPACE:
        # The action space only includes a non-discrete value in
        # a normalized range [-1, 1]. Actual range is [-u_high,u_high].
        self.action_space = spaces.Box(low=-1, high=1, 
                                       shape=(1,), dtype=np.float32)
        
        # OBSERVATION SPACE:   
        # Six observations normalized within [-1,1]
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(6,), dtype=np.float32)

    def step(self, u): 	
       
        # Current state (normalized between [-1,1]):
        th0, dth0, th1, dth1, th2, dth2 = self.state

        # Undo normalization to input state into ODE solver
        u_action = self.u_high * u    # N
        th0 = self.xmax_bd * th0      # m
        dth0 = self.spd_bd * dth0     # m/sec
        th1 = self.the1_bd * th1      # rad
        dth1 = self.angVel_bd * dth1  # rad/sec
        th2 = self.the2_bd * th2      # rad
        dth2 = self.angVel_bd * dth2  # rad/sec
                
        # self.prev_actions.append(u)
             
        # Calculate ODE solution 
        self.sol = odeSol( [th0, dth0, 
                            th1, dth1, 
                            th2, dth2], u_action)
        
        # Next state:
        self.state = self.sol[num_calcs - 1]
        th0 = self.state[0]
        dth0 = self.state[1]
        th1 = angle_normalize(self.state[2])
        dth1 = self.state[3]
        th2 = angle_normalize(self.state[4])
        dth2 = self.state[5]
        
              
        # Check if pendulum is inverted:
        if invert_check(self.time_held, self.hold_time_goal, th1, th2) == 2:
            self.time_held = self.time_held + self.t_action
            self.inv = True
        elif invert_check(self.time_held, self.hold_time_goal, th1, th2) == 1:
            self.time_held = self.time_held + self.t_action
        elif invert_check(self.time_held, self.hold_time_goal, th1, th2) == 0:
            self.time_held = 0
            
        
        self.time_elapsed = self.time_elapsed + self.t_action
        
       
        # Reward:
        amp = 2.
        shift = 0.85
        # For starting inverted
        # self.reward = (amp * (np.cos(np.pi * th0 / self.xmax_bd) - shift)/10
        #                # - dth0**2/100
        #                - (th1**2  + th2**2)/1.
        #                - (dth1**2 + dth2**2)/1.
        #                - u_action**2/4000
        #                + (self.time_held*1.)*1000.
        #                + self.time_elapsed*500.
        #                )
        
        self.reward = (amp * (np.cos(np.pi * th0 / self.xmax_bd) - shift)/100
                        - dth0**2/1000
                        - (th1**2  + th2**2)*10
                        - (dth1**2 + dth2**2)*10
                        - u_action**2/2000
                        + (self.time_held*1.)*400.
                        + self.time_elapsed*250.
                        )
                
        if (out_of_bounds(th0, self.xmax_bd) == 1 
            # or timeout(self.time_elapsed) == 1
            or speed_Lim(dth0, self.spd_bd) == 1
            or angVel_Lim(dth1, dth2, self.angVel_bd) == 1
            or abs(th1) >= np.pi/2
            or abs(th2) >= np.pi/2):
            self.done = True
            
        # Re-do normalization of state for observation: 
        self.state = [th0, dth0, th1, dth1, th2, dth2]
        self.state = np.array(self.state,dtype=np.float32) / np.array([self.xmax_bd, self.spd_bd,
                                                                       self.the1_bd, self.angVel_bd,
                                                                       self.the2_bd, self.angVel_bd], dtype=np.float32)
        
        if self.done:
            self.reward = self.reward - 350.
            
        if self.inv:
            self.reward = self.reward + 700.
        
        info = {}
    
                     
        return self._get_obs().astype(np.float32), float(self.reward), self.done, info
    
    
    def reset(self):
              
        # Initializing DPIC conditions
        self.u = 0
        self.time_elapsed = 0
        self.time_held = 0
        
        # Array of 4 random values between [-1, 1]
        rand_vals = 2 * np.random.rand(4) - 1
        
        # Initial state of DPIC
        self.cart_pos = 0.
        self.cart_speed = 0.
        self.pen1_ang = rand_vals[0]*.02
        self.pen1_angVel = 0.
        self.pen2_ang = rand_vals[0]*.02
        self.pen2_angVel = 0.
        
        # Initial state in single list
        self.state = np.array([self.cart_pos, self.cart_speed, 
                               self.pen1_ang, self.pen1_angVel, 
                               self.pen2_ang, self.pen2_angVel], dtype=np.float32)
        
        self.prev_reward = 0.
        self.done = False
        self.inv = False
                
        return self._get_obs()  # reward, done, info can't be included
    
    def _get_obs(self):
        theta, thetadot, theta1, theta1dot, theta2, theta2dot = self.state
        return np.array([theta, thetadot, 
                         theta1, theta1dot, 
                         theta2, theta2dot], 
                         dtype=np.float32)