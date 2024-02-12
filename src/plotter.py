import numpy as np
import scipy.special as ssp
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, resolution=500, velocity_bounds=[-3,3]):
        plt.ion()
        self.resolution = resolution
        self.velocity_bounds = velocity_bounds
        self.im = plt.imshow(np.zeros([1,1]))
        self.v = np.linspace(velocity_bounds[0], velocity_bounds[1], resolution)

    def plot1V(self, sys, species_ids, delay=1e-1):
        plt.clf()
        vals = np.zeros([sys.domain['N'], self.resolution])

        for s in species_ids:
            if (sys.species[s].num_modes>1).sum() != 1: # Counts how many velocity axes have more than 1 mode, if multiple have more than 1 mode, then this is not a 1V case and can´t be drawn as 2D plot
                raise Exception("Species ", s, " not 1V!")

            velocity_axis = np.argmax(sys.species[s].num_modes>1) # Find which velocity axis is the relevant one (i.e. the only one with >1 modes)
            n = np.arange(sys.species[s].num_modes[velocity_axis])

            xi = (self.v - sys.species[s].u[velocity_axis]) / sys.species[s].alpha[velocity_axis]

            C_s = sys.all_C(s)            
            for x in range(sys.domain['N']):
                if (velocity_axis==0):
                    C_x = C_s[:,0,0,x]
                elif (velocity_axis==1):
                    C_x = C_s[0,:,0,x]
                elif (velocity_axis==2):
                    C_x = C_s[0,0,:,x]
                
                # if x==0 and s==0: print(C_s[:,:,:,x])
                # exit()
                # print(np.power(2, n, dtype=float))
                # print(ssp.factorial(n))
                # print((np.power(2, n, dtype=float) * ssp.factorial(n)))
                # exit()
                
                C_x_HF = (np.power(2, n, dtype=float) * ssp.factorial(n) * np.pi**0.5)**-0.5 * C_x # Include (2^n * n! * sqrt(pi))^-0.5 term in coefficients
                # if x==0 and s==0: print(C_x_HF)
                
                vals[x,:] += np.polynomial.hermite.hermval(xi, C_x_HF) * np.exp(-xi**2) # Evaluate Hermite function series
        
        plt.imshow(vals.T, origin='lower', extent=[sys.domain['L'], sys.domain['R'], self.velocity_bounds[0], self.velocity_bounds[1]])
        plt.colorbar()
                        
        plt.pause(delay)