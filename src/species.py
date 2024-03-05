import numpy as np

class Species:
    def __init__(self, Z, m, u, alpha, collision_rate, num_modes, bc, initial):
        self.q = Z # Charge number
        self.m = m # Dimensionless mass
        self.shift = np.array(u, dtype=float)
        self.scale = np.array(alpha, dtype=float)
        self.collision_rate = collision_rate
        self.num_modes = np.array(num_modes)

        # BCs
        bc_types = [bc[0]['type'], bc[1]['type']]
        if ('P' in bc_types and 'D' in bc_types) or ('P' in bc_types and 'N' in bc_types):
            raise Exception("Can't mix periodic and Dirichlet/Neumann BC!")
        self.bc = bc
        self.initial = initial