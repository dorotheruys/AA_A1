import numpy as np
import matplotlib.pyplot as plt

# Define the conditions
Vinf = 10.  # [m/s]
AoA = 0.1  # [rad]

AoA_min = -0.5  # [rad]
AoA_max = 0.5  # [rad]
n_angles = 100  # [-]
AoA_range = np.linspace(AoA_min, AoA_max, n_angles)  # [rad]

rho = 1.225  # [kg/m3]
chord = 0.25  # [m]
n_sections = 1000  # [-]

chord_discr = np.linspace(0, chord, n_sections)  # [m], gets list of chord coordinates

def get_theta_from_x(chord, n_sections):
    # Inputs:
    # list or array of x locations (not undimensionized with chord length [m]
    # chord length of airfoil
    # Output:
    # Array of theta points [rad]
    chord_discr = np.linspace(0,chord,n_sections)
    theta = np.arccos(1-2*chord_discr/chord)
    return theta, chord_discr

def get_local_vorticity(AoA, Vinf, thetas):
    # Inputs:
    # list or array of theta locations [rad]
    # angle of attack [rad]
    # free stream velocity [m/s]
    # Output:
    # Array of vorticity at theta points
    vorticities = np.zeros(len(thetas))
    vorticities[1:] = 2 * Vinf * AoA * (1 + np.cos(thetas[1:])) / np.sin(thetas[1:])
    return vorticities

def lift_calc(rho, Vinf, vorticity, chord, thetas):
    # Inputs:
    # Vorticity distribution, Vinf, rho, chord, array of theta values
    # Outputs:
    # lift per unit span, lift coefficient
    lprime = 0.5 * chord * rho * Vinf * np.trapz(vorticity * np.sin(thetas), thetas)
    C_l = lprime / (0.5 * rho * Vinf * Vinf * chord)
    return lprime, C_l


def moment_calc(rho, Vinf, vorticity, chord, thetas):
    # Inputs:
    # density, freestream velocity, array of vorticity distribution, chordlength, array of theta distribution
    # Outputs:
    # moment coefficient per unit span, moment coefficient
    Mprime_LE = -0.25 * chord * chord * rho * Vinf * np.trapz(vorticity * (1 - np.cos(thetas)) * np.sin(thetas), thetas)
    C_M_LE = Mprime_LE / (0.5 * rho * Vinf * Vinf * chord * chord)
    return Mprime_LE, C_M_LE


def delta_Cp_calc(Vinf, vorticity):
    # Code to obtain delta-cp at the individual stations of the airfoil, using the freestream velocity and the vorticity distribution
    return 2 * vorticity / Vinf

def plot_Cl_alpha(AoA_range, chord, n_sections, Vinf, rho):
    thetas = get_theta_from_x(chord, n_sections)[0]

    Cl_range = []

    for AoA in AoA_range:
        vorticities = get_local_vorticity(AoA, Vinf, thetas)
        lift = lift_calc(rho, Vinf, vorticities, chord, thetas)
        Cl_range.append(lift[1])

    ref_Clalpha = 2 * np.pi * AoA_range
    delta = Cl_range - ref_Clalpha
    plt.plot(AoA_range, Cl_range, label="$\mathregular{C_{l}}$ using thin-airfoil theory")
    plt.plot(AoA_range, ref_Clalpha, label="Reference $\mathregular{C_{l}}$ with slope of 2π")
    plt.plot(AoA_range, delta, label="Delta between approximation and reference")
    plt.xlabel(r'$ \alpha $ [rad]')
    plt.ylabel("$\mathregular{C_{l}}$ [-]")
    plt.grid()
    plt.legend()
    plt.show()


def plot_Cm_alpha(AoA_range, chord, n_sections, Vinf, rho):
    thetas = get_theta_from_x(chord, n_sections)[0]

    Cm_range = []

    for AoA in AoA_range:
        vorticities = get_local_vorticity(AoA, Vinf, thetas)
        moments = moment_calc(rho, Vinf, vorticities, chord, thetas)
        Cm_range.append(moments[1])

    ref_Cmalpha = -0.5 * np.pi * AoA_range
    delta = Cm_range - ref_Cmalpha
    plt.plot(AoA_range, Cm_range, label="$\mathregular{C_{m_{LE}}}$ using thin-airfoil theory")
    plt.plot(AoA_range, ref_Cmalpha, label="Reference $\mathregular{C_{m_{LE}}}$ with slope of -0.5π")
    plt.plot(AoA_range, delta, label="Delta between approximation and reference")
    plt.xlabel(r'$ \alpha $ [rad]')
    plt.ylabel("$\mathregular{C_{m_{LE}}}$ [-]")
    plt.grid()
    plt.legend()
    plt.show()


def plot_Cp(AoA, chord, n_sections, Vinf):
    thetas, chord_discr = get_theta_from_x(chord, n_sections)
    vorticities = get_local_vorticity(AoA, Vinf, thetas)
    delta_Cp = delta_Cp_calc(Vinf, vorticities)
    plt.plot(chord_discr, delta_Cp, label="$\mathregular{C_{p}}$ distribution")
    plt.xlabel("x [m]")
    plt.ylabel("$\mathregular{C_{p}}$ [-]")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_Cl_alpha(AoA_range, chord, n_sections, Vinf, rho)
    # plot_Cm_alpha(AoA_range, chord, n_sections, Vinf, rho)
    plot_Cp(AoA, chord, n_sections, Vinf)
