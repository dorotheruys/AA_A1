import numpy as np
import matplotlib.pyplot as plt

# Define the conditions
Vinf = 10.  # [m/s]
Pinf = 101325  # [Pa]
AoA = np.deg2rad(5)  # [rad]

AoA_min = np.deg2rad(-10)  # [rad]
AoA_max = np.deg2rad(10)  # [rad]
n_angles = 10  # [-]
AoA_range = np.linspace(AoA_min, AoA_max, n_angles)  # [rad]

rho = 1.225  # [kg/m3]
chord = 1.  # [m]
n_sections = 100  # [-]

exp_AoA = np.deg2rad(4.18)
exp_data_chord = np.array(
    [0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.125, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.8,
     0.85, 0.9, 0.95])
exp_data_cpu_4 = np.array(
    [-3.1, -2.41, -1.82, -1.35, -0.93, -0.76, -0.64, -0.52, -0.46, -0.42, -0.37, -0.33, -0.3, -0.26, -0.22, -0.2, -0.17,
     -0.14, -0.07, 0.4, 0.06, 0.])
exp_data_cpl_4 = np.array(
    [0.55, 0.95, 0.77, 0.64, 0.39, 0.29, 0.23, 0, 0.14, 0, 0.07, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.,
     0.06, 0.])
exp_data_deltacp_4 = exp_data_cpu_4 - exp_data_cpl_4


def theta_discr(n_sections):
    # Function to obtain an array of values for theta between 0 and pi
    theta_discr = np.linspace(0, np.pi, n_sections)
    return theta_discr


def theta2(thetas):
    # Function to find values of theta between the points of the input, avoiding placing vorticity right at the LE and TE
    theta2 = np.zeros(len(thetas) - 1)
    for i in range(len(theta2)):
        theta2[i] = (thetas[i] + thetas[i + 1]) / 2
    return theta2


def x_discr(thetas, chord):
    # Function to do coordinate transformation from an array of thetas to an array of x-coordinates between 0 and chordlength
    return chord / 2 * (1 - np.cos(thetas))


def get_local_vorticity(AoA, Vinf, thetas):
    # Inputs:
    # list or array of theta locations [rad]
    # angle of attack [rad]
    # free stream velocity [m/s]
    # Output:
    # Array of vorticity at theta points
    vorticities = 2 * Vinf * AoA * (1 + np.cos(thetas)) / np.sin(thetas)
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
    return -2 * vorticity / Vinf


def Cp_upper_and_lower(Vinf, Pinf, rho, vorticity):
    P_u = 0.5 * rho * (Vinf - vorticity / 2)
    P_l = 0.5 * rho * (Vinf + vorticity / 2)

    Cp_u = (P_u - Pinf) / (0.5 * Vinf * Vinf * rho)
    Cp_l = (P_l - Pinf) / (0.5 * Vinf * Vinf * rho)

    return Cp_u, Cp_l


def plot_Cl_alpha(AoA_range, chord, n_sections, Vinf, rho):
    thetas = theta_discr(n_sections)
    thetas_new = theta2(thetas)

    Cl_range = []

    for AoA in AoA_range:
        vorticities = get_local_vorticity(AoA, Vinf, thetas_new)
        lift = lift_calc(rho, Vinf, vorticities, chord, thetas_new)
        Cl_range.append(lift[1])

    ref_Clalpha = 2 * np.pi * AoA_range
    delta = Cl_range - ref_Clalpha
    AoA_range = np.rad2deg(AoA_range)

    plt.plot(AoA_range, Cl_range, label="$\mathregular{C_{l}}$ using thin-airfoil theory")
    plt.plot(AoA_range, ref_Clalpha, label="Reference $\mathregular{C_{l}}$ with slope of 2$\pi$")
    plt.plot(AoA_range, delta, label="Delta between approximation and reference")
    plt.xlabel(r'$ \alpha $ [deg]')
    plt.ylabel("$\mathregular{C_{l}}$ [-]")
    plt.grid()
    plt.legend()
    plt.show()


def plot_Cm_alpha(AoA_range, chord, n_sections, Vinf, rho):
    thetas = theta_discr(n_sections)
    thetas_new = theta2(thetas)

    Cm_range = []

    for AoA in AoA_range:
        vorticities = get_local_vorticity(AoA, Vinf, thetas_new)
        moments = moment_calc(rho, Vinf, vorticities, chord, thetas_new)
        Cm_range.append(moments[1])

    ref_Cmalpha = -0.5 * np.pi * AoA_range
    delta = Cm_range - ref_Cmalpha
    AoA_range = np.rad2deg(AoA_range)
    plt.plot(AoA_range, Cm_range, label="$\mathregular{C_{m_{LE}}}$ using thin-airfoil theory")
    plt.plot(AoA_range, ref_Cmalpha, label="Reference $\mathregular{C_{m_{LE}}}$ with slope of -0.5$\pi$")
    plt.plot(AoA_range, delta, label="Delta between approximation and reference")
    plt.xlabel(r'$ \alpha $ [deg]')
    plt.ylabel("$\mathregular{C_{m_{LE}}}$ [-]")
    plt.grid()
    plt.legend()
    plt.show()


def plot_Cp(AoA, chord, n_sections, Vinf):
    thetas = theta_discr(n_sections)
    thetas_new = theta2(thetas)
    chord_discr = x_discr(thetas_new, chord)
    vorticities = get_local_vorticity(AoA, Vinf, thetas_new)
    delta_Cp = delta_Cp_calc(Vinf, vorticities)
    AoA_deg = np.rad2deg(AoA)
    plt.plot(chord_discr, delta_Cp, label=r' Code AoA = {:0.2f} [-]'.format(AoA_deg), color='red')


def read_xfoil_file(name):
    f = open(name, 'r')
    lines = f.readlines()
    dataupper = []
    datalower = []
    for l in lines[1:]:
        values = l.split()
        if len(values) == 3:
            if float(values[1]) > 0:
                dataupper.append([float(val) for val in values])
            elif float(values[1]) < 0:
                datalower.append([float(val) for val in values])
    f.close()

    dataupper_array = np.array(dataupper)
    datalower_array = np.array(datalower)

    return dataupper_array, datalower_array


def plot_xfoil_data(AoA_deg, xfoil_upper, xfoil_lower):
    deltaCp_data = []
    chord_distr = []
    for line in xfoil_upper:
        x = line[0]
        y = line[1]
        Cp_upper = line[2]

        index = np.where(xfoil_lower[:, 0] == x)
        if index[0].size > 0:
            Cp_lower = xfoil_lower[index, 2][0]

        deltaCp = Cp_upper - Cp_lower
        deltaCp_data.append(deltaCp)
        chord_distr.append(x)

    plt.plot(chord_distr, deltaCp_data, label=r'NACA0001 Xfoil AoA = {:0.2f} [-]'.format(AoA_deg), color='blue')

    return


if __name__ == "__main__":
    # plot_Cl_alpha(AoA_range, chord, n_sections, Vinf, rho)
    # plot_Cm_alpha(AoA_range, chord, n_sections, Vinf, rho)

    max_AoA_deg = (input('Max Angle of Attack [deg]? '))

    # Discretize the theta in a linear way
    thetas = theta_discr(n_sections)
    thetas_new = theta2(thetas)

    for AoA_deg in range(1, int(max_AoA_deg) + 1, 1):
        AoA = np.deg2rad(AoA_deg)
        vorticities = get_local_vorticity(AoA, Vinf, thetas_new)
        C_l = lift_calc(rho, Vinf, vorticities, chord, thetas_new)
        C_m = moment_calc(rho, Vinf, vorticities, chord, thetas_new)

        print('Lift coefficient C_l: ', C_l[1])
        print('Moment coefficient C_m: ', C_m[1])

        dataupper_array, datalower_array = read_xfoil_file(f'cp-aoa{AoA_deg}')

        plot_xfoil_data(AoA_deg, dataupper_array, datalower_array)
        plot_Cp(AoA, chord, n_sections, Vinf)

    plt.xlabel("x [m]")
    plt.ylabel("$\Delta\mathregular{C_{p}}$ [-]")
    plt.gca().invert_yaxis()
    plt.ylim((1, -4))
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()

    plot_Cp(exp_AoA, chord, n_sections, Vinf)
    plt.plot(exp_data_chord, exp_data_deltacp_4, label=r'NACA0006 Exp. AoA = {:0.2f} [-]'.format(np.rad2deg(exp_AoA)))

    plt.xlabel("x [m]")
    plt.ylabel("$\Delta\mathregular{C_{p}}$ [-]")
    plt.gca().invert_yaxis()
    plt.ylim((1, -4))
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


