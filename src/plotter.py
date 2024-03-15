import numpy as np
import scipy.special as ssp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

class Plotter:
    def __init__(self, save_directory=None, v_resolution=500, velocity_bounds=[-3,3]):
        self.v_resolution = v_resolution
        self.velocity_bounds = velocity_bounds
        self.im = plt.imshow(np.zeros([1,1]))
        self.v = np.linspace(velocity_bounds[0], velocity_bounds[1], v_resolution)
        
        Path(save_directory).mkdir(parents=True, exist_ok=True) # Create directory if it does not yet exist´
        self.save_directory = save_directory
        print(self.save_directory)



    def plot1V(self, sys, species_ids, t=None, E=None):
        plt.clf()
        gridspec = {'width_ratios': [20, 1], 'height_ratios': [3,1,1]}
        fig, axs = plt.subplots(3, 2, sharex=True, gridspec_kw=gridspec, figsize=(8,8))
        vals = np.zeros([sys.domain['N'], self.v_resolution])

        if t is not None: fig.suptitle(f"t = {t:.3f}")

        # Plot VDF & shift parameter
        for s in species_ids:
            if (sys.species[s].num_modes>1).sum() != 1: # Counts how many velocity axes have more than 1 mode, if multiple have more than 1 mode, then this is not a 1V case and can´t be drawn as 2D plot
                raise Exception("Species ", s, " not 1V!")

            velocity_axis = np.argmax(sys.species[s].num_modes>1) # Find which velocity axis is the relevant one (i.e. the only one with >1 modes)
            n = np.arange(sys.species[s].num_modes[velocity_axis])    

            C_s = sys.all_C(s)            
            for z_idx in range(sys.domain['N']):
                if (velocity_axis==0):
                    C_x = C_s[:,0,0,z_idx]
                elif (velocity_axis==1):
                    C_x = C_s[0,:,0,z_idx]
                elif (velocity_axis==2):
                    C_x = C_s[0,0,:,z_idx]
                
                C_x_HF = (np.power(2, n, dtype=float) * ssp.factorial(n) * np.pi)**-0.5 * C_x # Include (2^n * n! * sqrt(pi))^-0.5 term in coefficients
                xi = (self.v - sys.species[s].shift[velocity_axis, z_idx]) / sys.species[s].scale[velocity_axis, z_idx]
                vals[z_idx,:] += np.polynomial.hermite.hermval(xi, C_x_HF) * np.exp(-xi**2) # Evaluate Hermite function series
        
        im = axs[0,0].imshow(vals.T, origin='lower', extent=[sys.domain['L'], sys.domain['R'], self.velocity_bounds[0], self.velocity_bounds[1]], cmap="jet", aspect="auto")
        axs[0,0].set_ylabel(r"v")
        axs[0,0].set_title(f"VDF & Shift parameter")
        plt.colorbar(im, cax=axs[0,1])



        z = np.linspace(sys.domain['L'], sys.domain['R'], sys.domain['N'], endpoint=False)
        # Draw electric field
        if E is not None:
            axs[1,0].plot(z, E)
        axs[1,0].set_title(f"Electric field")
        axs[1,1].axis('off')

        # Draw scale parameter
        for s in species_ids:
            velocity_axis = np.argmax(sys.species[s].num_modes>1) # Find which velocity axis is the relevant one (i.e. the only one with >1 modes)
            color = next(axs[0,0]._get_lines.prop_cycler)['color']
            shift = sys.species[s].shift[velocity_axis]
            scale = sys.species[s].scale[velocity_axis]
            if sys.species[s].bc[0]["type"] == 'P' or sys.species[s].bc[1]["type"] == 'P':
                z_s = np.append(z, sys.domain['R'])
                shift = np.append(shift, shift[0])
                scale = np.append(scale, scale[0])
            axs[0,0].plot(z_s, shift, color=color, label=f"Species {s}")
            axs[2,0].plot(z_s, scale, color=color)
        axs[2,0].set_title(f"Scale parameter")
        axs[2,1].axis('off')

        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(handles))

        if self.save_directory is not None: plt.savefig(f"{self.save_directory}/t{t:.4f}_vdf.png")    

    def plot_coefficients(self, sys, s, modes, normalize=False, t=None):
        plt.clf()
        for i,j,k in modes:
            vals = sys.C(s,i,j,k)
            vals_max, vals_min = vals.max(), vals.min()
            if normalize and vals_max!=vals_min:
                vals = (vals - vals_min) / (vals_max - vals_min)
            plt.plot(vals, label=f"$C_{{{i},{j},{k}}}$, max={vals_max}")
        plt.gca().set_xlabel(r"z")
        plt.legend()
        if t is not None: plt.gca().set_title(f"t = {t:.3f}")
        if self.save_directory is not None: plt.savefig(f"{self.save_directory}/t{t:.4f}_coefficients.png")
