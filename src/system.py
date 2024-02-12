import numpy as np
from species import Species

class System():
    def __init__(self, species, domain):
        self.num_species = len(species)
        self.species = species
        self.domain = domain
        self.domain['hz'] = (self.domain['R'] - self.domain['L']) / self.domain['N']
        
        # Find total system size and create start/end index vectors
        self.Cn_s_idx = np.zeros((self.num_species, 2), dtype=int) # Start and end indices for every species in current timestep DOF vector
        self.Cnp1_s_idx = np.zeros((self.num_species, 2), dtype=int) # Start and end indices for every species in next timestep DOF vector
        system_size = 0
        for s in range(len(species)): # Iterate over all species
            self.Cn_s_idx[s, 0] = system_size # Store start index
            s_num_modes = self.species[s].num_modes.prod() # Number of modes for species s
            system_size += s_num_modes * self.domain['N'] # Add number of DOFs of current species to system size
            self.Cn_s_idx[s, 1] = system_size # Store end index

        self.Cn = np.zeros(system_size) # Allocate current timestep DOF vector
        self.Cnp1 = np.zeros(system_size) # Allocate next timestep DOF vector

        # Set initial condition with supplied initial condition function
        z = np.linspace(self.domain['L'], self.domain['R'], self.domain['N'])
        for s in range(len(self.species)): # Iterate over species, fill Cn for every species
            for i in range(self.species[s].num_modes[0]):
                for j in range(self.species[s].num_modes[1]):
                    for k in range(self.species[s].num_modes[2]):
                        idx = self.indices(s, i, j, k, prev=True)
                        self.Cn[idx[0]:idx[1]] = self.species[s].initial(i, j, k, z)


    def C(self, s, i, j, k, prev=False): # Field C_ijk
        # Return zeroes if i or j or k are out of range
        ijk = np.array([i, j, k])
        num_modes = self.species[s].num_modes
        if not np.logical_and(0<=ijk, ijk<num_modes).all(): # Return zero if i, j or k fall out of range
            return 0
        
        idx = self.indices(s, i, j, k, prev) # Compute indices into DOF vector for s, i, j, k
        return self.Cn[idx[0]:idx[1]] if prev else self.Cnp1[idx[0]:idx[1]]

    def dC(self, s, i, j, k, u, prev=False): # Spatial gradient of field C_ijk via central differences
        ijk = np.array([i, j, k])
        num_modes = self.species[s].num_modes
        if not np.logical_and(0<=ijk, ijk<num_modes).all(): # Return zero if i, j or k fall out of range
            return 0
        
        field = self.C(s, i, j, k, prev) # Retrieve field
        bc = self.species[s].bc

        if bc[0]['type'] == 'P' and bc[1]['type'] == 'P': # Periodic BCs
            # Upwind differences
            # if u > 0: # If velocity is in +z, then upwind is in -z direction
            #     diff = 1/(self.domain['hz']) * (np.roll(field, 1) - field)
            # else: # If velocity is in -z, then upwind is in +z direction
            #     diff = 1/(self.domain['hz']) * (field - np.roll(field, -1))
            # print(diff)
            diff = 1/(2*self.domain['hz']) * (np.roll(field, 1) - np.roll(field, -1))
            return diff * u
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
            return diff * u

    def indices(self, s, i, j, k, prev=False):
        s_idx = self.Cn_s_idx[s] #if prev else self.Cnp1_s_idx[s] # Get start and end indices for species s
        num_modes_i, num_modes_j, num_modes_k = self.species[s].num_modes # Get max number of i, j, k modes

        # print('-----')
        # print([i, j, k])
        # print(self.species[s].num_modes)
        start_idx = s_idx[0] + ((num_modes_i * num_modes_j) * k + num_modes_i * j + i) * self.domain['N'] # Compute start index for s, i, j, k
        end_idx = start_idx + self.domain['N'] # Compute end index from start index
        # print([start_idx, end_idx])
        # print('-----')   
        return [start_idx, end_idx]
    
    def all_C(self, s):
        s_idx = self.Cn_s_idx[s]
        num_modes = self.species[s].num_modes
        C_list = self.Cn[s_idx[0]:s_idx[1]]

        return C_list.reshape((num_modes[0], num_modes[1], num_modes[2], -1)) # Reshape such that coefficients are returned as 4D array with axes i, j, k, spatial