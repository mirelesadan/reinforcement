# Double Pendulum in Cart 
# Approximations and Animation 

# by Adan J. Mireles Fall 2022

import sympy as smp

# Defining corresponding symbols for each variable:

t, g = smp.symbols('t g')                       # Time and acceleratino due to gravity
m0, m1, m2, M = smp.symbols('m0 m1 m2 M')       # Mass of cart, pendulums, and mass of system
l1, l2, L1, L2 = smp.symbols('l1, l2, L1, L2')  # Length of rods (L) and distance from pendulum to center of mass of rod (l)
I1, I2 = smp.symbols('I1 I2')                   # Moments of inertia
u = smp.symbols('u')                            # Controller force

the0, the1, the2 = smp.symbols(r'\theta_0, \theta_1, \theta_2', cls=smp.Function)

# The following assumes that the center of mass of the rods is at the middle of their length:
l1 = L1/2
l2 = L2/2
I1 = m1*L1**2/12
I2 = m2*L2**2/12

the0 = the0(t) # Position of cart
the1 = the1(t) # Angle between pendulum 1 and the vertical 
the2 = the2(t) # Angle between pendulum 2 and the vertical 

the0_d = smp.diff(the0, t) # Velocity of cart
the1_d = smp.diff(the1, t) # Angular velocity of pendulum 1
the2_d = smp.diff(the2, t) # Angular velocity of pendulum 2

the0_dd = smp.diff(the0_d, t) # Acceleration of cart
the1_dd = smp.diff(the1_d, t) # Angular acceleration of pendulum 1
the2_dd = smp.diff(the2_d, t) # Angular acceleration of pendulum 2


a, b, c, d, e, f, G, h, i, j, k, l = smp.symbols('a, b, c, d, e, f, G, h, i, j, k, l')

# Lagrange Equations:
LE0 = a*the0_dd + b*the1_dd + c*the2_dd + d
LE1 = e*the0_dd + f*the1_dd + G*the2_dd + h
LE2 = i*the0_dd + j*the1_dd + k*the2_dd + l
sols = smp.solve([LE0,LE1, LE2],(the0_dd, the1_dd, the2_dd))

# Coefficients of each Lagrange equation, manually typed:

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

# The next step is to obtain uncoupled expressions for the second derivative of each theta.
# Using the three commented lines below (sol_theXdd = ...) will yield the same results as those in the succeeding lines.
# Note that using the eval() is not a good practice. 
# See: https://stackoverflow.com/questions/1832940/why-is-using-eval-a-bad-practice

# sol_the0dd = eval(str(sols[the0_dd]))
# sol_the1dd = eval(str(sols[the1_dd]))
# sol_the2dd = eval(str(sols[the2_dd]))

# A second way to obtain the results is to print each of the results below, and copy the output into code manually.
sol_the0dd = sols[the0_dd] # Print and assign manually below
sol_the1dd = sols[the1_dd] # Print and assign manually below
sol_the2dd = sols[the2_dd] # Print and assign manually below

# = [Manually inserted below via Copy-Paste]
sol_the0dd = (-G*b*l + G*d*j + b*h*k + c*f*l - c*h*j - d*f*k)/(-G*a*j + G*b*i + a*f*k - b*e*k + c*e*j - c*f*i)
sol_the1dd = (G*a*l - G*d*i - a*h*k - c*e*l + c*h*i + d*e*k)/(-G*a*j + G*b*i + a*f*k - b*e*k + c*e*j - c*f*i)
sol_the2dd = (-a*f*l + a*h*j + b*e*l - b*h*i - d*e*j + d*f*i)/(-G*a*j + G*b*i + a*f*k - b*e*k + c*e*j - c*f*i)

# Here, note that lambdify converts symbolic expressions into numerical functions.

dz0dt_f = smp.lambdify((t,u,g,m0,m1,m2,L1,L2,the0,the1,the2,the0_d,the1_d,the2_d), sol_the0dd)
dz1dt_f = smp.lambdify((t,u,g,m0,m1,m2,L1,L2,the0,the1,the2,the0_d,the1_d,the2_d), sol_the1dd)
dz2dt_f = smp.lambdify((t,u,g,m0,m1,m2,L1,L2,the0,the1,the2,the0_d,the1_d,the2_d), sol_the2dd)
dthe0dt_f = smp.lambdify(the0_d, the0_d)
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)

