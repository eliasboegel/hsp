import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')))
import hsp

dt = 0.1
t_final = 100
n0 = 0.5
v_th = 1/np.sqrt(8)
alpha = np.sqrt(2) * v_th
u_avg_1 = 1
u_avg_2 = -1
n_points = 50
n_modes = 25
shift_adaptivity = "xt"
scale_adaptivity = "xt"
integrator = hsp.Time.implicit_euler
save_directory = f"../images/tsi_{n_points}z_{n_modes}vz_u-{shift_adaptivity}f_a-{scale_adaptivity}f_plottest"


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
p = hsp.Plotter(save_directory=save_directory)

ts = dt * np.arange(1, int(t_final/dt) + 1)
p.plot1V(s.sys, [0,1], t=0, E=s.E)
E_norm = np.zeros_like(ts)
for t_idx in range(ts.shape[0]):
    t = ts[t_idx]
    print(f't = {t:.3f}')
    s.step(dt)
    s.adapt(shift_mode=shift_adaptivity, scale_mode=scale_adaptivity)
    E_norm[t_idx] = np.linalg.norm(s.E, 1) # Record 1-norm of electric field
    p.plot1V(s.sys, [0,1], t=t, E=s.E)

# Plot electric field norm evolution over time
plt.clf()
plt.plot(ts, E_norm)
plt.gca().set_yscale("log")
plt.gca().set_xlabel(r"$t$ [-]")
plt.gca().set_ylabel(r"$|E|_1$ [-]")
plt.savefig(f"{save_directory}/E_norm.png")
plt.show()