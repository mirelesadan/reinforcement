# Double Pendulum in Cart 
# Approximations and Animation 

#    by Adan J. Mireles 10/18/22

# Import necessary packages
# In[1]:
import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy.parsing.sympy_parser import parse_expr

# Defining corresponding symbols for each variable:
# Note that an extra mass is included (m0, the mass of cart)

# In[2]:


t, g = smp.symbols('t g')
m0, m1, m2, M = smp.symbols('m0 m1 m2 M')
l1, l2, L1, L2 = smp.symbols('l1, l2, L1, L2')
I1, I2 = smp.symbols('I1 I2')
u = smp.symbols('u')         # controller force


# In[3]:

the0, the1, the2 = smp.symbols(r'\theta_0, \theta_1, \theta_2', cls=smp.Function)


# In[4]

# Position, angles, and their rates of change    

the0 = the0(t)
the1 = the1(t)
the2 = the2(t)

the0_d = smp.diff(the0, t)
the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the0_dd = smp.diff(the0_d, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)


# In[6]:


a, b, c, d, e, f, G, h, i, j, k, l = smp.symbols('a, b, c, d, e, f, G, h, i, j, k, l')

LE0 = a*the0_dd + b*the1_dd + c*the2_dd + d
LE1 = e*the0_dd + f*the1_dd + G*the2_dd + h
LE2 = i*the0_dd + j*the1_dd + k*the2_dd + l
sols = smp.solve([LE0,LE1, LE2],(the0_dd, the1_dd, the2_dd))


# This assumes that the center of mass of the rods is at the middle of their length:

# In[7]:


l1 = L1/2
l2 = L2/2
I1 = m1*L1**2/12
I2 = m2*L2**2/12


# In[8]: Coefficients of each lagrange equation

# From Lagrange Equation 1
a = m0 + m1 + m2
b = (m1*l1 + m2*L1)*smp.cos(the1)
c = m2*l2*smp.cos(the2)
d = -(m1*l1 + m2*L1)*smp.sin(the1)*the1_d**2 - m2*l2*smp.sin(the2)*the2_d**2 - u

# From Lagrange Equation 2
e = (m1*l1 + m2*L1)*smp.cos(the1)
f = m1*l1**2 + m2*L1**2 + I1
G = m2*L1*l2*smp.cos(the1-the2)
h = m2*L1*l2*smp.sin(the1-the2)*the2_d**2 - (m1*l1 + m2*L1)*g*smp.sin(the1)

# From Lagrange Equation 3
i = m2*l2*smp.cos(the2)
j = m2*L1*l2*smp.cos(the1-the2)
k = m2*l2**2 + I2
l = -m2*L1*l2*smp.sin(the1-the2)*the1_d**2 - m2*g*l2*smp.sin(the2)


# It is necessary to evaluate the values $(a,b,c,\cdots,k,l)$ in the $\ddot{\theta}_i$ equations.

# In[9]:

# eval() built-in function may led to security issues
# sol_the0dd = eval(str(sols[the0_dd]))
# sol_the1dd = eval(str(sols[the1_dd]))
# sol_the2dd = eval(str(sols[the2_dd]))

sol_the0dd = sols[the0_dd] # Print and assign manually below
sol_the1dd = sols[the1_dd] # Print and assign manually below
sol_the2dd = sols[the2_dd] # Print and assign manually below

sol_the0dd = (-G*b*l + G*d*j + b*h*k + c*f*l - c*h*j - d*f*k)/(-G*a*j + G*b*i + a*f*k - b*e*k + c*e*j - c*f*i)
sol_the1dd = (G*a*l - G*d*i - a*h*k - c*e*l + c*h*i + d*e*k)/(-G*a*j + G*b*i + a*f*k - b*e*k + c*e*j - c*f*i)
sol_the2dd = (-a*f*l + a*h*j + b*e*l - b*h*i - d*e*j + d*f*i)/(-G*a*j + G*b*i + a*f*k - b*e*k + c*e*j - c*f*i)

# The "**odeint**" solver requires that the maximum order of the differential equation is 1. 
# Hence, we define the following as functions that take inputs:

# <center> $z_i=\dot{\theta_i}$ </center>
# <center> $\dot{z_i}=\ddot{\theta_i}$ </center>

# In[10]:

# Here, note that lambdify converts symbolic expressions into numerical functions.

dz0dt_f = smp.lambdify((t,u,g,m0,m1,m2,L1,L2,the0,the1,the2,the0_d,the1_d,the2_d), sol_the0dd)
dz1dt_f = smp.lambdify((t,u,g,m0,m1,m2,L1,L2,the0,the1,the2,the0_d,the1_d,the2_d), sol_the1dd)
dz2dt_f = smp.lambdify((t,u,g,m0,m1,m2,L1,L2,the0,the1,the2,the0_d,the1_d,the2_d), sol_the2dd)
dthe0dt_f = smp.lambdify(the0_d, the0_d)
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)

# Moreover, the "**odeint**" solver also requires that we compress the initial 
# conditions $(\theta_i,z_i)$, as well as those for the properties of the 
# pendulum and cart, into a single quantity $\vec{S}$. 
# This enables the solver to compute $\frac{d\vec{S}}{dt}$

# In[12]:


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

dSdt([1,2,3,4,5,6],2,3,4,5,6,7,8,9)
# Now defining initial conditions:  

# In[13]:


t = np.linspace(0, 40, 1001)  # 1000/40 = 25 calculations per second
g = 9.81 #m/s^2
m0 = 2.1 #kg
m1= 0.7 #kg
m2= 0.7 #kg
L1 = 0.8 #m
L2 = 0.8 #m


# Using the "**odeint**" solver:

# Here, $y_0$ $=$ [$\theta_0$, $\dot{\theta_0}$, $\theta_1$, $\dot{\theta_1}$, $\theta_2$, $\dot{\theta_2}$]:

# In[24]:

# Action ...
u = 0  # Control force (F = m_cart*a)
ans = odeint(dSdt, y0=[0, 0, np.pi/2, 0, np.pi/2, 0], 
             t=t, args=(u, g, m0, m1, m2, L1, L2))

the0 = ans.T[0]
the1 = ans.T[2]
the2 = ans.T[4]


# In[25]:

#plt.plot(t, the0, label='$\\theta_0$')
#plt.plot(t, the1, label='$\\theta_1$')
#plt.plot(t, the2, label='$\\theta_2$')
#plt.legend()
#plt.xlabel('Time (sec)')
#plt.ylabel('$\\theta_i$ (rad) for $i>0$, $(m)$ otherwise')


# Each row in the "**ans**" matrix yields the $(\theta_0,\dot{\theta_0},\theta_1,\dot{\theta_1},\theta_2,\dot{\theta_2})$ state at time $t$

# In[23]:
ans


# Upon **transposing**, we obtain a matrix of 6 rows corresponding to the values of each variable mentioned above.

# In[144]:
ans.T

# Having the above data, it is possible to track the location of the pendulums using what we used in **In [18]**

# In[95]:

def xy_coords(t, the0, the1, the2, L1, L2):
    return (the0,
            0*the0,
            the0 + L1*np.sin(the1),
            L1*np.cos(the1),
            the0 + L1*np.sin(the1) + L2*np.sin(the2),
            L1*np.cos(the1) + L2*np.cos(the2))

x0, y0, x1, y1, x2, y2 = xy_coords(t, ans.T[0], ans.T[2], ans.T[4], L1, L2)
