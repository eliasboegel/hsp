import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')))
from solver import Solver
from species import Species
from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt

domain = {
    'L': 0, # Left boundary position
    'R': 2*np.pi, # Right boundary position
    'N': 100 # Number of points
}

def perturbed_maxwellian1(i, j, k, z):
    if i==0 and j==0 and k==0:
        return 0.5*(1 + 1e-2*np.cos(2*np.pi*z/(domain['R'] - domain['L']))) # Coefficient for first mode = Maxwellian
    else:
        return 0
    
def perturbed_maxwellian2(i, j, k, z):
    if i==0 and j==0 and k==0:
        return 0.5*(1) # Coefficient for first mode = Maxwellian
    else:
        return 0

species = [ # Arguments in order: charge number, atomic mass [u], u, alpha, collison_rate, ijk_max, BCs, initial_condition
    Species(-1, 1, [0,0,1], [0.3,0.3,0.3], 5, [1,1,300], # Singly charged Xenon ions with 1 radial mode, 1 azimuth mode, 4 axial modes
    [
        {'type': 'P'}, # Left BC
        {'type': 'P'}, # Right BC
    ], perturbed_maxwellian1),
    Species(-1, 1, [0,0,-1], [0.3,0.3,0.3], 5, [1,1,300], # Singly charged Xenon ions with 1 radial mode, 1 azimuth mode, 4 axial modes
    [
        {'type': 'P'}, # Left BC
        {'type': 'P'}, # Right BC
    ], perturbed_maxwellian2)
]

E_bc = {
    'type': 'P',
    #'values': [300, 0] # Potential at left and right boundary if Dirichlet (type 'D') BC is used
}

s = Solver(domain, species, E_bc)
p = Plotter()

p.plot1V(s.sys, [0,1], delay=1)
# plt.ion()
# plt.plot(s.sys.C(0,0,0,49,prev=True))
# initial = s.sys.C(0,0,0,49,prev=True)
# plt.pause(3)
dt = 1
t = 0
for i in range(10000):
    s.step(dt)
    t += dt
    print(f't = {t:.3f}')
    p.plot1V(s.sys, [0,1])
    # plt.clf()
    # plt.plot(initial)
    # plt.plot(s.sys.C(0,0,0,49,prev=True))
    # plt.pause(1e-1)
    
    

