import numpy as np

import forcesmoments as fm
import vorticity as vort

# Define the conditions
Vinf = 10.  # [m/s]
AoA = 0.1  # [rad]
# AoA_range = np.linspace(-0.5,0.5,10) #[rad]
rho = 1.225  # [kg/m3]
chord = 0.25  # [m]

chord_discr = np.linspace(0, chord, 5)  # [m], gets list of chord coordinates

if __name__ == "__main__":
    thetas = vort.get_theta_from_x(chord_discr, chord)
    vorticities = vort.get_local_vorticity(AoA, Vinf, thetas)

    lift = fm.lift(rho, Vinf, vorticities, chord, thetas)
    moments = fm.moments(rho, Vinf, vorticities, chord, thetas)

    delta_Cp = fm.delta_Cp(Vinf,vorticities)
    print(thetas)
    print(delta_Cp)