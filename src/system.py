import numpy as np

class System(): # Class to manage DOFs in a single vector compatible with SciPy JFNK format
    def __init__(self, species, domain):
        self.num_species = len(species)
        self.species = species
        self.domain = domain
        self.domain['hz'] = (self.domain['R'] - self.domain['L']) / self.domain['N']
        
        # Find total system size and create start/end index vectors
        self.dof_idx = np.zeros((self.num_species, 2), dtype=int) # Start and end indices for every species in current timestep DOF vector
        system_size = 0
        for s in range(len(species)): # Iterate over all species
            self.dof_idx[s, 0] = system_size # Store start index
            s_num_modes = self.species[s].num_modes.prod() # Number of modes for species s
            system_size += s_num_modes * self.domain['N'] # Add number of DOFs of current species to system size
            self.dof_idx[s, 1] = system_size # Store end index

        self.dof = np.zeros(system_size) # Allocate DOF vector

        # Set initial condition with supplied initial condition function
        z = np.linspace(self.domain['L'], self.domain['R'], self.domain['N'], endpoint=False)
        for s in range(len(self.species)): # Iterate over species, fill Cn for every species
            for i in range(self.species[s].num_modes[0]):
                for j in range(self.species[s].num_modes[1]):
                    for k in range(self.species[s].num_modes[2]):
                        idx = self.indices(s, i, j, k)
                        self.dof[idx[0]:idx[1]] = self.species[s].initial(i, j, k, z)


    def C(self, s, i, j, k): # Field C_ijk
        # Return zeroes if i or j or k are out of range
        ijk = np.array([i, j, k])
        num_modes = self.species[s].num_modes
        if not np.logical_and(0<=ijk, ijk<num_modes).all(): # Return zero if i, j or k fall out of range
            return 0
        
        idx = self.indices(s, i, j, k) # Compute indices into DOF vector for s, i, j, k
        vals = self.dof[idx[0]:idx[1]]
        return vals
        

    def dC(self, s, i, j, k): # Spatial gradient of field C_ijk via central differences
        ijk = np.array([i, j, k])
        num_modes = self.species[s].num_modes
        if not np.logical_and(0<=ijk, ijk<num_modes).all(): # Return zero if i, j or k fall out of range
            return 0
        
        Cn = self.C(s, i, j, k) # Retrieve field
        # Cnm2 = np.roll(Cn, 2)
        Cnm1 = np.roll(Cn, 1)
        Cnp1 = np.roll(Cn, -1)
        # Cnp2 = np.roll(Cn, -2)

        bc = self.species[s].bc

        if bc[0]['type'] == 'P' and bc[1]['type'] == 'P': # Periodic BCs
            diff = (Cnp1 - Cnm1) / (2*self.domain['hz']) # 2nd order central differences
            # diff = (-Cnp2 + 8*Cnp1 - 8*Cnm1 + Cnm2) / (12*self.domain['hz']) # 4th order central differences
            return diff
        else: # Dirichlet or Neumann BCs
            if bc[0]['type'] == 'D':
                l_val = bc[0]['values'](i, j, k)
            elif bc[0]['type'] == 'N':
                l_val = self.C(s, i, j, k)[0] - 2*self.domain['hz']*bc[0]['values'](i, j, k) # Compute appropriate ghost point value for Neumann condition

            if bc[1]['type'] == 'D':
                r_val = bc[1]['values'](i, j, k)
            elif bc[1]['type'] == 'N':
                r_val = self.C(s, i, j, k)[-1] + 2*self.domain['hz']*bc[1]['values'](i, j, k) # Compute appropriate ghost point value for Neumann condition

            padded_field = np.concatenate([[l_val], field, [r_val]])
            diff = 1/(2*self.domain['hz']) * (padded_field[2:] - padded_field[:-2])
            return diff

    def indices(self, s, i, j, k):
        s_idx = self.dof_idx[s] # Get start and end indices for species s
        num_modes_i, num_modes_j, num_modes_k = self.species[s].num_modes # Get max number of i, j, k modes

        start_idx = s_idx[0] + ((num_modes_i * num_modes_j) * k + num_modes_i * j + i) * self.domain['N'] # Compute start index for s, i, j, k
        end_idx = start_idx + self.domain['N'] # Compute end index from start index

        return [start_idx, end_idx]
    
    def all_C(self, s):
        s_idx = self.dof_idx[s]
        num_modes = self.species[s].num_modes
        C_list = self.dof[s_idx[0]:s_idx[1]]

        return C_list.reshape((num_modes[0], num_modes[1], num_modes[2], -1)) # Reshape such that coefficients are returned as 4D array with axes i, j, k, spatial

    def project_to(self, shift=None, scale=None, num_modes=None):
        new_species = self.species
        new_species.shift = shift
        new_species.scale = scale

        new_sys = System(new_species, self.domain)

        raise NotImplementedError("System projection not implemented!")

