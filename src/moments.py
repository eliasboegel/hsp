import numpy as np

class Moments():
    @staticmethod # Zero-order moment
    def number_density(sys, s):
        number_density = sys.species[s].scale.prod(axis=0) * sys.C(s,0,0,0)
        return number_density

    @staticmethod # Zero-order moment
    def charge_density(sys):
        # Charge density = sum of q_s * n_s, where q_s is the charge of the species and n_s is the number density of the species
        # number density of the species is directly related to the 0-th mode, i.e. n_s = alpha_1*alpha_2*alpha_3*C_0_0_0
        charge_density = np.zeros(sys.domain['N'])
        for s in range(len(sys.species)): # Iterate over species, add equations for every species
            number_density = Moments.number_density(sys, s)
            charge_density += sys.species[s].q * number_density # Add contribution of species to total charge density
        return charge_density

    @staticmethod # First-order moment
    def mean_velocity(sys, s):
        mean_velocity = np.empty((3, sys.domain['N']))
        mean_velocity[0] = sys.species[s].shift[0] + np.sqrt(1/2) * sys.species[s].scale[0] * sys.C(s, 1, 0, 0) / sys.C(s, 0, 0, 0)
        mean_velocity[1] = sys.species[s].shift[1] + np.sqrt(1/2) * sys.species[s].scale[1] * sys.C(s, 0, 1, 0) / sys.C(s, 0, 0, 0)
        mean_velocity[2] = sys.species[s].shift[2] + np.sqrt(1/2) * sys.species[s].scale[2] * sys.C(s, 0, 0, 1) / sys.C(s, 0, 0, 0)
        return mean_velocity

    @staticmethod # First-order moment
    def current_density(sys):
        raise NotImplementedError("Current density moment not yet implemented")

    @staticmethod # Second-order moment
    def pressure_tensor(sys, s):
        v_diff = sys.species[s].shift - Moments.mean_velocity(sys, s)
        pressure_tensor = np.zeros((3, 3, sys.domain['N']))
        # Fill diagonals
        pressure_tensor[0,0] = sys.species[s].scale.prod(axis=0) * sys.species[s].scale[0] / np.sqrt(2) * (sys.C(s,2,0,0) + 2*v_diff[0]*sys.C(s,1,0,0) + (sys.species[s].scale[0]/np.sqrt(2) + v_diff[0]**2)*sys.C(s,0,0,0) )
        pressure_tensor[1,1] = sys.species[s].scale.prod(axis=0) * sys.species[s].scale[1] / np.sqrt(2) * (sys.C(s,0,2,0) + 2*v_diff[1]*sys.C(s,0,1,0) + (sys.species[s].scale[1]/np.sqrt(2) + v_diff[1]**2)*sys.C(s,0,0,0) )
        pressure_tensor[2,2] = sys.species[s].scale.prod(axis=0) * sys.species[s].scale[2] / np.sqrt(2) * (sys.C(s,0,0,2) + 2*v_diff[2]*sys.C(s,0,0,1) + (sys.species[s].scale[2]/np.sqrt(2) + v_diff[2]**2)*sys.C(s,0,0,0) )

        # TODO: Anisotropic components of pressure tensor
        # ...


        pressure_tensor *= sys.species[s].m
        

        return pressure_tensor

    @staticmethod # Second-order moment
    def pressure_scalar(sys, s):
        pressure_tensor = Moments.pressure_tensor(sys, s)
        scalar_pressure = 1/3 * np.trace(pressure_tensor, axis1=0, axis2=1)
        return scalar_pressure
    
    @staticmethod # Second-order moment
    def temperature_tensor(sys, s):
        number_density = Moments.number_density(sys, s)
        pressure_tensor = Moments.pressure_tensor(sys, s)
        temperature_tensor = pressure_tensor / number_density # TODO: Include Boltzmann constant

    @staticmethod # Second-order moment
    def temperature_scalar(sys, s):
        number_density = Moments.number_density(sys, s)
        pressure_scalar = Moments.pressure_scalar(sys, s)
        temperature_scalar = pressure_scalar / number_density # TODO: Include Boltzmann constant
        return temperature_scalar

    @staticmethod # Second-order moment
    def thermal_velocity_tensor(sys, s):
        number_density = Moments.number_density(sys, s)
        pressure_tensor = Moments.pressure_tensor(sys, s)
        thermal_velocity_tensor = np.sqrt( pressure_tensor / (sys.species[s].m * number_density) )
        return thermal_velocity_tensor

    @staticmethod # Second-order moment
    def thermal_velocity_scalar(sys, s):
        number_density = Moments.number_density(sys, s)
        scalar_pressure = Moments.pressure_scalar(sys, s)
        thermal_velocity_scalar = np.sqrt( scalar_pressure / (sys.species[s].m * number_density) )
        return thermal_velocity_scalar