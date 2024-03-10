from copy import deepcopy
import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import matplotlib.pyplot as plt

from species import Species
from system import System
from operators import *
from moments import *

class Solver:
    def __init__(self, domain, species, integrator, E_bc):
        # Initialize system and set initial shift and scaling parameters for all species
        self.sys = System(species, domain)
        self.integrator = integrator
        self.E_bc = E_bc

    def solve_poisson(self, sys, periodic=False):
        # Using centered finite differences, with Dirichlet BC, potential(x=0) = potential [V], potential(x=L) = 0 [V]
        # If periodic BCs are not used, then use Dirichlet BC with given potential applied

        # Compute charge density
        charge_density = Moments.charge_density(self.sys)

        # LHS using lowest order central differences
        hz = (sys.domain['R'] - sys.domain['L']) / sys.domain['N']
        e0 = 8.8541878128e-12 # Vacuum permittivity [F/m]
        a0 = -2*np.ones(sys.domain["N"]) # Values for main diagonal
        a1 = np.ones(sys.domain["N"]-1) # Values for -1st and 1st diagonals
        b = hz**2 * charge_density # RHS
        
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
            E = 1/(2*hz) * (np.roll(pot, -1) - np.roll(pot, 1))
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
                        idx = sys.indices_spatial(s, i, j, k) # Get indices of components for given species and mode

                        root_vector[idx] = self.integrator(
                            self.sys, # Current timestep system (as held in Solver object as current state)
                            sys, # Current JFNK inner iteration estimate for system at next timestep
                            KineticEquation.cartesian, # Kinetic equation operator to use
                            dt, # Time step
                            s, i, j, k, E=self.E, B=0 # Mode information and EM fields
                        )

        return root_vector # Return root vector

    def adapt(self, shift_threshold=0.01, scale_threshold=0.01):
        new_species = deepcopy(self.sys.species)

        # Compute current mean velocity and temperature
        # Change species shift and scaling parameter to new mean velocity & temperature
        # Project old system to new system

        for s in range(len(self.sys.species)):

            # Compute current spatial average of current mean velocity
            mean_velocity = Moments.mean_velocity(self.sys, s)
            candidate_shift = mean_velocity
            
            # Smooth candidate shift
            # candidate_shift = gaussian_filter1d(candidate_shift, self.sys.domain['N']/25, axis=1, mode="wrap")
            # print(candidate_shift.shape)


            # candidate_shift[0] = candidate_shift[0].mean()
            # candidate_shift[1] = candidate_shift[1].mean()
            # candidate_shift[2] = candidate_shift[2].mean()
            # print(candidate_shift.shape)

            # print(f"s: {s}, max: {candidate_shift[2].max()}, min: {candidate_shift[2].min()}")

            # TODO: Compute current spatial average of temperature and candidate scale
            # ...
            thermal_velocity_tensor = Moments.thermal_velocity_tensor(self.sys, s)
            candidate_scale = np.empty_like(candidate_shift)
            candidate_scale[0] = np.sqrt(2) * thermal_velocity_tensor[0,0]
            candidate_scale[1] = np.sqrt(2) * thermal_velocity_tensor[1,1]
            candidate_scale[2] = np.sqrt(2) * thermal_velocity_tensor[2,2]
            candidate_scale[0] = candidate_scale[0].mean()
            candidate_scale[1] = candidate_scale[1].mean()
            candidate_scale[2] = candidate_scale[2].mean()

            # Set new basis if substantially candidate parameters are different from current parameters and perform solution projection
            for i in range(3):
                shift_adapt_mask = np.abs(new_species[s].shift[i] - candidate_shift[i]) > shift_threshold
                scale_adapt_mask = np.abs(new_species[s].scale[i] - candidate_scale[i]) > scale_threshold
                if np.any(shift_adapt_mask):
                    new_species[s].shift[i] = candidate_shift[i]
                    print(f"Shift parameter adapted for species {s} in axis {i}")
                if np.any(scale_adapt_mask):
                    new_species[s].scale[i] = candidate_scale[i]
                    print(f"Scale parameter adapted for species {s} in axis {i}")
                
        self.sys.project(new_species) # Project system to new basis

    def step(self, dt):
        # Set up new system to be modified in each JFNK inner iteration
        # When adaptivity in time is introduced, then write projection from one System to another
        sys = deepcopy(self.sys) # Copy old state, works provided no modes are added or removed for next timestep

        # Call SciPy JFNK implementation, store result in current timestep system
        self.sys.dof = newton_krylov(lambda x: self.nonlinear_system(x, sys, dt), sys.dof,
                                    method='lgmres',
                                    x_rtol=1e-8,
                                    verbose=True)

