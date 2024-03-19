import numpy as np
import scipy.special as ssp

class System(): # Class to manage DOFs in a single vector compatible with SciPy JFNK format
    def __init__(self, species, domain):
        system_size = domain['N'] * np.sum([s.num_modes.prod() for s in species])
        self.species = species
        self.shifts = np.zeros((len(species), 3, domain['N']))
        self.scales = np.zeros((len(species), 3, domain['N']))
        self.domain = domain
        self.dof = np.zeros(system_size) # Allocate DOF vector
        self.dof_idx = self.indices_species(species) # Compute and store start and end indices for each species
        self.set_hermite_parameters(species) # Set shift and scale parameters for each species
        self.set_initial_condition(species) # Set initial condition with supplied initial condition function

    def set_hermite_parameters(self, species):
        for s in range(len(species)): # Iterate over species, set shift and scale parameters for each
            # Set shift and scale parameter
            for axis in [0,1,2]:
                self.shifts[s,axis] = species[s].shift[axis]
                self.scales[s,axis] = species[s].scale[axis]

    def set_initial_condition(self, species):
        z = np.linspace(self.domain['L'], self.domain['R'], self.domain['N'], endpoint=False)

        for s in range(len(species)): # Iterate over species, fill Cn for every species
            for i in range(self.species[s].num_modes[0]):
                for j in range(self.species[s].num_modes[1]):
                    for k in range(self.species[s].num_modes[2]):
                        idx = self.indices_spatial(s, i, j, k)
                        self.dof[idx] = self.species[s].initial(i, j, k, z)

    def shift(self, s):
        return self.shifts[s]

    def shift_grad(self, s):
        hz = (self.domain['R'] - self.domain['L']) / self.domain['N']
        shift_nm1 = np.roll(self.shifts[s], 1, axis=1)
        shift_np1 = np.roll(self.shifts[s], -1, axis=1)
        shift_grad = (shift_np1 - shift_nm1) / (2*hz) # 2nd order FD
        return shift_grad

    def scale(self, s):
        return self.scales[s]

    def scale_grad(self, s):
        hz = (self.domain['R'] - self.domain['L']) / self.domain['N']
        scale_nm1 = np.roll(self.scales[s], 1, axis=1)
        scale_np1 = np.roll(self.scales[s], -1, axis=1)
        scale_grad = (scale_np1 - scale_nm1) / (2*hz) # 2nd order FD
        return scale_grad

    def C(self, s, i, j, k): # Field C_ijk
        # Return zeroes if i or j or k are out of range
        ijk = np.array([i, j, k])
        num_modes = self.species[s].num_modes
        if not np.logical_and(0<=ijk, ijk<num_modes).all(): # Return zero if i, j or k fall out of range
            return 0
        
        idx = self.indices_spatial(s, i, j, k) # Compute indices into DOF vector for s, i, j, k
        vals = self.dof[idx]
        return vals
        
    def C_grad(self, s, i, j, k): # Spatial gradient of field C_ijk via central differences
        hz = (self.domain['R'] - self.domain['L']) / self.domain['N']

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
            diff = (Cnp1 - Cnm1) / (2*hz) # 2nd order central differences
            # diff = (-Cnp2 + 8*Cnp1 - 8*Cnm1 + Cnm2) / (12*hz) # 4th order central differences
            return diff
        else: # Dirichlet or Neumann BCs
            if bc[0]['type'] == 'D':
                l_val = bc[0]['values'](i, j, k)
            elif bc[0]['type'] == 'N':
                l_val = self.C(s, i, j, k)[0] - 2*hz*bc[0]['values'](i, j, k) # Compute appropriate ghost point value for Neumann condition

            if bc[1]['type'] == 'D':
                r_val = bc[1]['values'](i, j, k)
            elif bc[1]['type'] == 'N':
                r_val = self.C(s, i, j, k)[-1] + 2*hz*bc[1]['values'](i, j, k) # Compute appropriate ghost point value for Neumann condition

            padded_field = np.concatenate([[l_val], field, [r_val]])
            diff = 1/(2*hz) * (padded_field[2:] - padded_field[:-2])
            return diff

    def indices_species(self, species):
        # Find total system size and create start/end index vectors
        species_idx = np.zeros((len(species), 2), dtype=int) # Start and end indices for every species in current timestep DOF vector
        system_size = 0
        for s in range(len(species)): # Iterate over all species
            species_idx[s, 0] = system_size # Store start index
            s_num_modes = species[s].num_modes.prod() # Number of modes for species s
            system_size += s_num_modes * self.domain['N'] # Add number of DOFs of current species to system size
            species_idx[s, 1] = system_size # Store end index
        
        return species_idx

    def indices_spatial(self, s, i, j, k):
        s_idx = self.dof_idx[s] # Get start and end indices for species s
        num_modes_i, num_modes_j, num_modes_k = self.species[s].num_modes # Get max number of i, j, k modes

        start_idx = s_idx[0] + ((num_modes_i * num_modes_j) * k + num_modes_i * j + i) * self.domain['N'] # Compute start index for s, i, j, k
        end_idx = start_idx + self.domain['N'] # Compute end index from start index

        return np.arange(start_idx, end_idx)
    
    def indices_mode(self, s, z_idx):
        s_idx = self.dof_idx[s] # Get start and end indices for species s
        num_modes_i, num_modes_j, num_modes_k = self.species[s].num_modes # Get max number of i, j, k modes

        return np.arange(s_idx[0], s_idx[1], self.domain['N']) + z_idx

    def all_C(self, s):
        s_idx = self.dof_idx[s]
        num_modes = self.species[s].num_modes
        C_list = self.dof[s_idx[0]:s_idx[1]]

        return C_list.reshape((num_modes[0], num_modes[1], num_modes[2], -1)) # Reshape such that coefficients are returned as 4D array with axes i, j, k, spatial

    def project(self, new_shift, new_scale): # Projects the system to a new basis

        # Different transformation must be applied for every spatial coordinate if spatial adaptivity is enabled
        # Therefore, loop over spatial coordinates
        z = np.linspace(self.domain['L'], self.domain['R'], self.domain['N'], endpoint=False)
        
        for s in range(len(self.species)):
            num_dof = self.species[s].num_modes.prod() # Current number of DOFs for any given spatial DOF
            transform = np.eye(num_dof, num_dof) # Basic transformation matrix if none of the parameters are changed, this is left-multiplied with all following transformation matrices

            # In the same ordering as the global DOF vector ordering, create vector containing i, j, k indices for all mode DOFs at a given spatial DOF
            i_1d = np.repeat(np.arange(self.species[s].num_modes[0]), self.species[s].num_modes[1]*self.species[s].num_modes[2])
            j_1d = np.repeat(np.tile(np.arange(self.species[s].num_modes[1]), self.species[s].num_modes[0]), self.species[s].num_modes[2])
            k_1d = np.tile(np.arange(self.species[s].num_modes[2]), self.species[s].num_modes[0]*self.species[s].num_modes[1])

            # Create matrices stacking the 1d index vectors into i,j,k (old DOFs) and n,m,p (new DOFs) index matrices, i.e. a matrix with a unique combination of i,j,k,n,m,p for every element
            a = np.tile(i_1d, (num_dof, 1))
            b = np.tile(j_1d, (num_dof, 1))
            c = np.tile(k_1d, (num_dof, 1))
            d = a.T
            e = b.T
            f = c.T

            # Apply transform to all coordinates
            for z_idx in range(self.domain['N']):
                # Extract relevant DOFs
                idx = self.indices_mode(s, z_idx) # Indices for DOFs of species s at given spatial index
                C_old = self.dof[idx]
                C_new = C_old # C_new is temporary state to which transformations are applied

                np.seterr(divide='ignore', invalid='ignore') # Next section intentionally contains divide by zero resulting in inf, which is overwritten with zeroes in all cases

                # Compute shift parameter transform
                shift_transform = np.sqrt( (ssp.factorial(d) * ssp.factorial(e) * ssp.factorial(f))/(ssp.factorial(a) * ssp.factorial(b) * ssp.factorial(c) * np.float_power(ssp.factorial(d-a), 2) * np.float_power(ssp.factorial(e-b), 2) * np.float_power(ssp.factorial(f-c), 2) ) * np.float_power(2, d+e+f-a-b-c) )
                shift_transform[d-a<0] = 0
                shift_transform[e-b<0] = 0
                shift_transform[f-c<0] = 0
                
                
                if np.any(self.shift(s)[0] != new_shift[s,0]): # If first shift parameter has changed
                    shift_transform_x = shift_transform * np.float_power( (self.shift(s)[0,z_idx] - new_shift[s,0,z_idx])/self.scale(s)[0,z_idx], d+e+f-a-b-c)
                    shift_transform_x = np.nan_to_num(shift_transform_x)
                    C_new = shift_transform_x @ C_new # Apply x-shift transform
                if np.any(self.shift(s)[1] != new_shift[s,1]): # If second shift parameter has changed
                    shift_transform_y = shift_transform * np.float_power( (self.shift(s)[1,z_idx] - new_shift[s,1,z_idx])/self.scale(s)[1,z_idx], d+e+f-a-b-c)
                    shift_transform_y = np.nan_to_num(shift_transform_y)
                    C_new = shift_transform_y @ C_new # Apply y-shift transform
                if np.any(self.shift(s)[2] != new_shift[s,2]): # If third shift parameter has changed
                    shift_transform_z = shift_transform * np.float_power( (self.shift(s)[2,z_idx] - new_shift[s,2,z_idx])/self.scale(s)[2,z_idx], d+e+f-a-b-c)
                    shift_transform_z = np.nan_to_num(shift_transform_z)
                    C_new = shift_transform_z @ C_new # Apply z-shift transform
                    if np.logical_not(np.isfinite(shift_transform_z)).sum() > 0:
                        np.set_printoptions(threshold=1000000)
                        print(shift_transform_z)

            
                # Compute shift parameter transform
                scale_transform = np.sqrt( (ssp.factorial2(d) * ssp.factorial2(e) * ssp.factorial2(f) * ssp.factorial2(d-1) * ssp.factorial2(e-1) * ssp.factorial2(f-1))/(ssp.factorial(a) * ssp.factorial(b) * ssp.factorial(c) * ssp.factorial2(d-a)**2 * ssp.factorial2(e-b)**2 * ssp.factorial2(f-c)**2 ) )
                scale_transform[d-a<0] = 0
                scale_transform[e-b<0] = 0
                scale_transform[f-c<0] = 0
                scale_transform[(a+d)%2!=0] = 0 # Set a+d odd to zero
                scale_transform[(b+e)%2!=0] = 0 # Set b+e odd to zero
                scale_transform[(c+f)%2!=0] = 0 # Set c+f odd to zero
                
                if np.any(self.scale(s)[0] != new_scale[s,0]): # If first scale parameter has changed
                    scale_transform_x = scale_transform * np.float_power(self.scale(s)[0,z_idx], a+b+c+1) / np.float_power(new_scale[s,0,z_idx], d+e+f+1) * np.float_power(self.scale(s)[0,z_idx]**2 - new_scale[s,0,z_idx]**2, (d+e+f-a-b-c)/2)
                    scale_transform_x = np.nan_to_num(scale_transform_x)
                    C_new = scale_transform_x @ C_new # Apply z-scale transform
                if np.any(self.scale(s)[1] != new_scale[s,1]): # If second scale parameter has changed
                    scale_transform_y = scale_transform * np.float_power(self.scale(s)[1,z_idx], a+b+c+1) / np.float_power(new_scale[s,1,z_idx], d+e+f+1) * np.float_power(self.scale(s)[1,z_idx]**2 - new_scale[s,1,z_idx]**2, (d+e+f-a-b-c)/2)
                    scale_transform_y = np.nan_to_num(scale_transform_y)
                    C_new = scale_transform_y @ C_new # Apply y-scale transform
                if np.any(self.scale(s)[2] != new_scale[s,2]): # If third scale parameter has changed
                    scale_transform_z = scale_transform * np.float_power(self.scale(s)[2,z_idx], a+b+c+1) / np.float_power(new_scale[s,2,z_idx], d+e+f+1) * np.float_power(self.scale(s)[2,z_idx]**2 - new_scale[s,2,z_idx]**2, (d+e+f-a-b-c)/2)
                    scale_transform_z = np.nan_to_num(scale_transform_z)
                    C_new = scale_transform_z @ C_new # Apply z-scale transform

                np.seterr(divide='warn', invalid='warn') # Reset warning level for div-by-zero

                self.dof[idx] = C_new # Set DOFs in new system
        self.shifts = new_shift
        self.scales = new_scale


        # # TODO: Add or remove Hermite modes
        # if np.any([np.any(self.species[s].num_modes != new_species[s].num_modes) for s in range(len(self.species))]): # Check if mode number was changed for any species
        #     new_system_size = self.domain['N'] * np.sum([s.num_modes.prod() for s in new_species])
        #     new_dof = np.zeros(new_system_size) # Allocate new DOF vector
        #     new_dof_idx = self.indices_species(new_species) # Compute and store start and end indices for each species

        #     # TODO: Move DOFs from old to new vector
        #     for s in range(len(new_species)):
        #         for i in range(self.species[s].num_modes[0]):
        #             for j in range(self.species[s].num_modes[1]):
        #                 for k in range(self.species[s].num_modes[2]):
        #                     # Uses same logic as indices_spatial() but for new species and new DOF vector
        #                     s_idx = new_dof_idx[s] # Get start and end indices for species s
        #                     num_modes_i, num_modes_j, num_modes_k = new_species[s].num_modes # Get max number of i, j, k modes
        #                     start_idx = s_idx[0] + ((num_modes_i * num_modes_j) * k + num_modes_i * j + i) * self.domain['N'] # Compute start index for s, i, j, k
        #                     end_idx = start_idx + self.domain['N'] # Compute end index from start index
        #                     idx = np.arange(start_idx, end_idx)

        #                     # Write DOFs to new DOF vector
        #                     # If new basis has fewer modes, then the cut off modes are removed
        #                     # If new basis has more modes, then the remaining ones stay initialized with zeroes as the mode iteration doesn't reach all new modes
        #                     new_dof[idx] = self.C(s, i, j, k)
                            
        #     # Write new DOFs and DOF indices to system
        #     self.species = new_species
        #     self.dof = new_dof
        #     self.dof_idx = new_dof_idx
