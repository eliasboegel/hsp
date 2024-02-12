import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov, InverseJacobian, BroydenFirst
import matplotlib.pyplot as plt

from species import Species
from system import System

class Solver:
    def __init__(self, domain, species, E_bc):
        self.sys = System(species, domain)
        self.E_bc = E_bc



    def solve_poisson(self, charge_density, periodic=False):
        # Using centered finite differences, with Dirichlet BC, potential(x=0) = potential [V], potential(x=L) = 0 [V]
        # If periodic BCs are not used, then use Dirichlet BC with given potential applied

        # LHS using lowest order central differences

        # plt.plot(charge_density)
        # plt.show()

        e0 = 8.8541878128e-12 # Vacuum permittivity [F/m]
        a0 = -2*np.ones(self.sys.domain["N"]) # Values for main diagonal
        a1 = np.ones(self.sys.domain["N"]-1) # Values for -1st and 1st diagonals
        b = self.sys.domain['hz']**2 * charge_density * 1e2 # RHS
        
        if self.E_bc['type'] == 'P':
            # Tridiagonal matrix with additional single element on top right and bottom left for periodicity
            # Add only top right single element, effectively specifying Dirichlet BC on one boundary to make solution unique
            A = spa.diags(
                [[1], a1, a0, a1, [1]],
                offsets=[-self.sys.domain["N"]+1, -1, 0, 1, self.sys.domain["N"]-1],
                format='csc'
            )
            b = b - b.mean() # It is necessary to fix int(b)dz = 0 to make the periodic BC system well conditioned

            pot = spsolve(A, b)
            E = 1/(2*self.sys.domain['hz']) * (np.roll(pot, -1) - np.roll(pot, 1))
            return E
        elif self.E_bc['type'] == 'D':
            A = spa.diags(
                [a1, a0, a1],
                offsets=[-1,0,1],
                format='csc'
            )
            b[0] -= self.E_bc['values'][0]
            b[-1] -= self.E_bc['values'][1]
        
            # Solution vector & solve
            pot = np.empty(self.sys.domain["N"] + 2) # Introduce full potential vector including Dirichlet BC
            pot[0] = self.E_bc['values'][0]
            pot[-1] = self.E_bc['values'][1]
            pot[1:-1] = spsolve(A, b) # Replace inner values with solution of Poisson equation (solved potential)

            # There is a mismatch in point locations for the E-field compared to the potential. We want to evaluate the E-field (as -grad(pot)) at the same points as where pot is defined.
            # To solve this, take FD gradient on the left and right of each point and take the mean
            E = -(np.diff(pot[:-1]) + np.diff(pot[1:])) / 2
            return E
        
    def LHS(self, s, i, j, k, Ez):
        # Implementation of the nonlinear LHS operator of the kinetic equation for species s and mode i,j,k
        
        # print([i,j,k])
        terms = 0
        # print(terms)
        terms += self.sys.dC(s, i, j, k,
                             self.sys.species[s].u[2]
        )
        # print(terms.mean())
        terms += self.sys.dC(s, i, j, k-1,
                             np.sqrt(k/2) * self.sys.species[s].alpha[2]
        )
        # print(terms.mean())
        terms += self.sys.dC(s, i, j, k+1, 
                             np.sqrt((k+1)/2) * self.sys.species[s].alpha[2]
        )
        # print(terms.mean())
        # print(self.sys.C(s, i, j, k-1))
        # print(Ez.mean())
        # print(self.sys.C(s, i, j, k-1))
        terms += - Ez * self.sys.species[s].q/self.sys.species[s].m * np.sqrt(2*k) / self.sys.species[s].alpha[2] * self.sys.C(s, i, j, k-1)
        # print(terms.mean())

        return terms

    def RHS(self, s, i, j, k):
        # Implementation of the RHS operator of the kinetic equation (collision operators) for species s and mode i,j,k
        Nr, Ntheta, Nz = self.sys.species[s].num_modes - 1
        coll_i = i*(i-1)*(i-2)/(Nr*(Nr-1)*(Nr-2)) if Nr>2 else 0
        coll_j = j*(j-1)*(j-2)/(Ntheta*(Ntheta-1)*(Ntheta-2)) if Ntheta>2 else 0
        coll_k = k*(k-1)*(k-2)/(Nz*(Nz-1)*(Nz-2)) if Nz>2 else 0
        return - self.sys.species[s].collision_rate * self.sys.C(s, i, j, k) * (coll_i + coll_j + coll_k)
        # return 0

    # Function that represents a nonlinear root finding problem with Cnp1 being the solution to f(Cnp1)=0 of the next timestep
    def nonlinear_system(self, Cnp1):
        # Allocate root vector, one component per species per mode per spatial point
        f_Cnp1 = np.empty_like(self.sys.Cnp1)
        
        self.sys.Cnp1 = Cnp1
        # print(self.sys.Cnp1)
        
        
        # Compute electric field via Poisson equation
        # The first iteration over species is necessary to evaluate the total charge density for computation of the E-field
        # Charge density = sum of q_s * n_s, where q_s is the charge of the species and n_s is the number density of the species
        # number density of the species is directly related to the 0-th mode, i.e. n_s = C_0_0_0
        charge_density = np.zeros(self.sys.domain['N'])
        for s in range(len(self.sys.species)): # Iterate over species, add equations for every species
            number_density = self.sys.species[s].alpha.prod() * self.sys.C(s, 0, 0, 0)
            charge_density += self.sys.species[s].q * number_density # Add contribution of species to total charge density
        Ez = self.solve_poisson(charge_density, periodic=True) # Compute E-field from charge density
        # print(Ez.mean())

        # Fill root vector one species, one mode at a time
        for s in range(len(self.sys.species)): # Iterate over species, add equations for every species
            # Iterate over all modes
            for i in range(self.sys.species[s].num_modes[0]):
                for j in range(self.sys.species[s].num_modes[1]):
                    for k in range(self.sys.species[s].num_modes[2]):
                        idx = self.sys.indices(s, i, j, k) # Get indices of components for given species and mode
                        # print([s, i, j, k], ': ', idx)
                        # Implicit Euler: dC/dt = K(C) -> C(t+dt) = C(t) + K(C(t+dt))*dt with K being a nonlinear operator representing the kinetic equation
                        K = self.RHS(s, i, j, k) - self.LHS(s, i, j, k, Ez) # Nonlinear operator representing kinetic equation (as LHS is nonlinear), K = dC/dt = RHS - LHS
                        
                        
                        f_Cnp1[idx[0]:idx[1]] = self.sys.Cnp1[idx[0]:idx[1]] - self.sys.Cn[idx[0]:idx[1]] - K * self.dt # Full nonlinear operator representing the implicit Euler method
                        # print('---------')
                        # print([s,i,j,k])
                        # print("x: ", self.sys.Cnp1[idx[0]:idx[1]].mean())
                        # print("f: ", f_Cnp1[idx[0]:idx[1]].mean())
        # print(f_Cnp1)
        # print('---------')
        # print("x: ", self.sys.Cnp1.mean())
        # print("|f|: ", np.linalg.norm(f_Cnp1))

        return f_Cnp1 # Return root vector

    def step(self, dt):
        self.dt = dt
        
        # sol = root(self.nonlinear_system, self.sys.Cn, method='krylov', callback=debug_callback)
        self.sys.Cn = newton_krylov(self.nonlinear_system, self.sys.Cn,
                                    method='lgmres',
                                    # inner_M=InverseJacobian(BroydenFirst()),
                                    verbose=True)
        