from copy import deepcopy
import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov
import matplotlib.pyplot as plt

from species import Species
from system import System
from operators import *

class Solver:
    def __init__(self, domain, species, integrator, E_bc):
        self.sys = System(species, domain)
        self.integrator = integrator
        self.E_bc = E_bc

    def solve_poisson(self, sys, periodic=False):
        # Using centered finite differences, with Dirichlet BC, potential(x=0) = potential [V], potential(x=L) = 0 [V]
        # If periodic BCs are not used, then use Dirichlet BC with given potential applied


        # Compute charge density
        # Charge density = sum of q_s * n_s, where q_s is the charge of the species and n_s is the number density of the species
        # number density of the species is directly related to the 0-th mode, i.e. n_s = alpha_1*alpha_2*alpha_3*C_0_0_0
        charge_density = np.zeros(sys.domain['N'])
        for s in range(len(sys.species)): # Iterate over species, add equations for every species
            number_density = sys.species[s].scale.prod() * sys.C(s, 0, 0, 0)
            charge_density += sys.species[s].q * number_density # Add contribution of species to total charge density


        # LHS using lowest order central differences
        e0 = 8.8541878128e-12 # Vacuum permittivity [F/m]
        a0 = -2*np.ones(sys.domain["N"]) # Values for main diagonal
        a1 = np.ones(sys.domain["N"]-1) # Values for -1st and 1st diagonals
        b = sys.domain['hz']**2 * charge_density # RHS
        
        if self.E_bc['type'] == 'P':
            # Tridiagonal matrix with additional single element on top right and bottom left for periodicity
            # Add only top right single element, effectively specifying Dirichlet BC on one boundary to make solution unique
            A = spa.diags(
                [[1], a1, a0, a1, [1]],
                offsets=[-sys.domain["N"]+1, -1, 0, 1, sys.domain["N"]-1],
                format='csc'
            )
            b = b - b.mean() # It is necessary to fix int(b)dz = 0 to make the periodic BC system well conditioned

            pot = spsolve(A, b)
            E = 1/(2*sys.domain['hz']) * (np.roll(pot, -1) - np.roll(pot, 1))
        elif self.E_bc['type'] == 'D':
            A = spa.diags(
                [a1, a0, a1],
                offsets=[-1,0,1],
                format='csc'
            )
            b[0] -= self.E_bc['values'][0]
            b[-1] -= self.E_bc['values'][1]
        
            # Solution vector & solve
            pot = np.empty(sys.domain["N"] + 2) # Introduce full potential vector including Dirichlet BC
            pot[0] = self.E_bc['values'][0]
            pot[-1] = self.E_bc['values'][1]
            pot[1:-1] = spsolve(A, b) # Replace inner values with solution of Poisson equation (solved potential)

            # There is a mismatch in point locations for the E-field compared to the potential. We want to evaluate the E-field (as -grad(pot)) at the same points as where pot is defined.
            # To solve this, take FD gradient on the left and right of each point and take the mean
            E = (np.diff(pot[:-1]) + np.diff(pot[1:])) / 2
        else:
            raise NotImplementedError("Chosen BC for Poisson equation is invalid or not implemented!")
        return E
        
    # Function that represents a nonlinear root finding problem with x being the solution to f(x)=0 of the next timestep
    def nonlinear_system(self, x, sys, dt):
        root_vector = np.zeros_like(sys.dof) # Allocate root vector, one component per species per mode per spatial point
        sys.dof = x # Set updated DOF values of current JFNK inner iteration

        # Compute electric field at timestep t+dt
        self.E = self.solve_poisson(sys, periodic=True) # Compute E-field from charge density

        # Fill root vector one species, one mode at a time
        for s in range(len(sys.species)): # Iterate over species, add equations for every species
            for i in range(sys.species[s].num_modes[0]):
                for j in range(sys.species[s].num_modes[1]):
                    for k in range(sys.species[s].num_modes[2]):
                        idx = sys.indices(s, i, j, k) # Get indices of components for given species and mode

                        root_vector[idx[0]:idx[1]] = self.integrator(
                            self.sys, # Current timestep system (as held in Solver object as current state)
                            sys, # Current JFNK inner iteration estimate for system at next timestep
                            KineticEquation.cartesian, # Kinetic equation operator to use
                            dt, # Time step
                            s, i, j, k, E=self.E, B=0 # Mode information and EM fields
                        )

        return root_vector # Return root vector

    def step(self, dt):
        # Set up new system to be modified in each JFNK inner iteration
        # When adaptivity in time is introduced, then write projection from one System to another
        sys = deepcopy(self.sys) # Copy old state, works provided no modes are added or removed for next timestep

        # Call SciPy JFNK implementation, store result in current timestep system
        self.sys.dof = newton_krylov(lambda x: self.nonlinear_system(x, sys, dt), sys.dof,
                                    method='lgmres',
                                    x_rtol=1e-8,
                                    verbose=True)

