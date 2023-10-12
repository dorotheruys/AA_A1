import numpy as np
import matplotlib.pyplot as plt

import forcesmoments as fm
import plots
import vorticity as vort
import plots as plots

# Define the conditions
Vinf = 10.  # [m/s]
AoA = 0.01  # [rad]

AoA_min = -0.5  # [rad]
AoA_max = 0.5  # [rad]
n_angles = 100  # [-]
AoA_range = np.linspace(AoA_min, AoA_max, n_angles)  # [rad]

rho = 1.225  # [kg/m3]
chord = 0.25  # [m]
n_sections = 250  # [-]

chord_discr = np.linspace(0, chord, n_sections)  # [m], gets list of chord coordinates

if __name__ == "__main__":
    plots.plot_Cl_alpha(AoA_range, chord, n_sections, Vinf, rho)
    plots.plot_Cm_alpha(AoA_range, chord, n_sections, Vinf, rho)
    plots.plot_Cp(AoA, chord, n_sections, Vinf)
