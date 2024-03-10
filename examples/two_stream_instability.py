import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')))
import hsp

dt = 0.1
t_final = 50
n0 = 0.5
v_th = 1/np.sqrt(8)
alpha = np.sqrt(2) * v_th
u_avg_1 = 1
u_avg_2 = -1
n_points = 50
n_modes = 25
integrator = hsp.Time.implicit_euler

save_directory = f"../images/tsi_{n_points}z_{n_modes}vz_u-xt_a-t"


domain = {
    'L': 0, # Left boundary position
    'R': 2*np.pi, # Right boundary position
    'N': n_points # Number of points
}
hz = (domain['R'] - domain['L']) / domain['N']

def perturbed_maxwellian(i, j, k, z):
    if i==0 and j==0 and k==0:
        return n0/alpha * (1 + 1e-1*np.cos(2*np.pi*z/(domain['R'] - domain['L'])))
    else:
        return 0

shift_initial_1 = np.zeros((3, domain['N']))
shift_initial_1[2] = u_avg_1
shift_initial_2 = np.zeros((3, domain['N']))
shift_initial_2[2] = u_avg_2
scale_initial = np.ones((3, domain['N']))
scale_initial[2] = alpha


species = [ # Arguments in order: charge number, mass, u, alpha, collison_rate, ijk_max, BCs, initial_condition
    hsp.Species(-1, 1, shift_initial_1, scale_initial, 5, [1,1,n_modes], # First beam
    [
        {'type': 'P'}, # Left BC
        {'type': 'P'}, # Right BC
    ], perturbed_maxwellian),
    hsp.Species(-1, 1, shift_initial_2, scale_initial, 5, [1,1,n_modes], # Second beam
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
p.plot1V(s.sys, [0,1], delay=1, t=t_idx*dt)
p.plot_coefficients(s.sys, 0, [[0,0,0], [0,0,n_modes//25], [0,0,n_modes//10], [0,0,n_modes//3], [0,0,n_modes-1]], normalize=True, delay=1, t=t_idx*dt)
while t_idx*dt < t_final:
    s.step(dt)
    s.adapt()
    t_idx += 1
    print(f't = {t_idx*dt:.3f}, t_idx = {t_idx}')
    p.plot1V(s.sys, [0,1], E=s.E, t=t_idx*dt)
    p.plot_coefficients(s.sys, 0, [[0,0,0], [0,0,n_modes//25], [0,0,n_modes//10], [0,0,n_modes//3], [0,0,n_modes-1]], normalize=True, t=t_idx*dt)
