import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')))
import hsp

dt = 0.05
t_final = 100
n0 = 0.5
v_th = 1/np.sqrt(8)
alpha = np.sqrt(2) * v_th
u_avg_1 = 1
u_avg_2 = -1
n_points = 199
n_modes = 250
integrator = hsp.Time.implicit_euler

save_directory = f"../images/tsi_{n_points}z_{n_modes}vz"


domain = {
    'L': 0, # Left boundary position
    'R': 2*np.pi, # Right boundary position
    'N': n_points # Number of points
}

def perturbed_maxwellian(i, j, k, z):
    if i==0 and j==0 and k==0:
        return n0/alpha * (1 + 1e-1*np.cos(2*np.pi*(z)/(domain['R'] - domain['L'] + domain['hz'])))
    else:
        return 0
    
def maxwellian(i, j, k, z):
    if i==0 and j==0 and k==0:
        return n0/alpha
    else:
        return 0

species = [ # Arguments in order: charge number, atomic mass [u], u, alpha, collison_rate, ijk_max, BCs, initial_condition
    hsp.Species(-1, 1, [0,0,u_avg_1], [1,1,alpha], 5, [1,1,n_modes], # Singly charged Xenon ions with 1 radial mode, 1 azimuth mode, 4 axial modes
    [
        {'type': 'P'}, # Left BC
        {'type': 'P'}, # Right BC
    ], perturbed_maxwellian),
    hsp.Species(-1, 1, [0,0,u_avg_2], [1,1,alpha], 5, [1,1,n_modes], # Singly charged Xenon ions with 1 radial mode, 1 azimuth mode, 4 axial modes
    [
        {'type': 'P'}, # Left BC
        {'type': 'P'}, # Right BC
    ], perturbed_maxwellian)
]

E_bc = {
    'type': 'P',
    #'values': [300, 0] # Potential at left and right boundary if Dirichlet (type 'D') BC is used
}

s = hsp.Solver(domain, species, integrator, E_bc)
p = hsp.Plotter(save_directory=save_directory, show=False)

t_idx = 0
p.plot1V(s.sys, [0,1], delay=0.5, t=t_idx*dt)
# p.plot_coefficients(s.sys, 0, [[0,0,0]], normalize=True, delay=1, t=t_idx*dt)
while t_idx*dt < t_final:
    s.step(dt)
    t_idx += 1
    print(f't = {t_idx*dt:.3f}, t_idx = {t_idx}')
    p.plot1V(s.sys, [0,1], E=s.E, t=t_idx*dt)
    # p.plot_coefficients(s.sys, 0, [[0,0,0]], normalize=True, t=t_idx*dt)
    