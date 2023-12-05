import numpy as np
import matplotlib.pyplot as plt

# Define the conditions
Vinf = 10.  # Free stream velocity in m/s
Pinf = 101325  # Free stream pressure in Pa
AoA = np.deg2rad(5)  # Angle of attack in radians

# Define the range of angles of attack
AoA_min = np.deg2rad(-10)  # Minimum angle of attack in radians
AoA_max = np.deg2rad(10)  # Maximum angle of attack in radians
n_angles = 10  # Number of angles to consider
AoA_range = np.linspace(AoA_min, AoA_max, n_angles)  # Range of angles of attack in radians

rho = 1.225  # Air density in kg/m3
chord = 1.  # Chord length in m
n_sections = 100  # Number of sections to divide the airfoil into

# Experimental data for comparison
exp_AoA = np.deg2rad(4.18)  # Experimental angle of attack in radians
exp_data_chord = np.array(
    [0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.125, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.8,
     0.85, 0.9, 0.95])  # Experimental data for chordwise pressure distribution
exp_data_cpu_4 = np.array(
    [-3.1, -2.41, -1.82, -1.35, -0.93, -0.76, -0.64, -0.52, -0.46, -0.42, -0.37, -0.33, -0.3, -0.26, -0.22, -0.2, -0.17,
     -0.14, -0.07, 0.4, 0.06, 0.])  # Experimental data for upper surface pressure coefficient
exp_data_cpl_4 = np.array(
    [0.55, 0.95, 0.77, 0.64, 0.39, 0.29, 0.23, 0, 0.14, 0, 0.07, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.,
     0.06, 0.])  # Experimental data for lower surface pressure coefficient
exp_data_deltacp_4 = exp_data_cpu_4 - exp_data_cpl_4  # Experimental data for pressure coefficient difference

# Function to obtain an array of values for theta between 0 and pi
def theta_discr(n_sections):
    theta_discr = np.linspace(0, np.pi, n_sections)
    return theta_discr

# Function to find values of theta between the points of the input, avoiding placing vorticity right at the LE and TE
def theta2(thetas):
    theta2 = np.zeros(len(thetas) - 1)
    for i in range(len(theta2)):
        theta2[i] = (thetas[i] + thetas[i + 1]) / 2
    return theta2

# Function to do coordinate transformation from an array of thetas to an array of x-coordinates between 0 and chordlength
def x_discr(thetas, chord):
    return chord / 2 * (1 - np.cos(thetas))

# Function to calculate local vorticity at theta points
def get_local_vorticity(AoA, Vinf, thetas):
    vorticities = 2 * Vinf * AoA * (1 + np.cos(thetas)) / np.sin(thetas)
    return vorticities

# Function to calculate lift per unit span and lift coefficient
def lift_calc(rho, Vinf, vorticity, chord, thetas):
    lprime = 0.5 * chord * rho * Vinf * np.trapz(vorticity * np.sin(thetas), thetas)
    C_l = lprime / (0.5 * rho * Vinf * Vinf * chord)
    return lprime, C_l

# Function to calculate moment coefficient per unit span and moment coefficient
def moment_calc(rho, Vinf, vorticity, chord, thetas):
    Mprime_LE = -0.25 * chord * chord * rho * Vinf * np.trapz(vorticity * (1 - np.cos(thetas)) * np.sin(thetas), thetas)
    C_M_LE = Mprime_LE / (0.5 * rho * Vinf * Vinf * chord * chord)
    return Mprime_LE, C_M_LE

# Function to calculate delta-cp at the individual stations of the airfoil
def delta_Cp_calc(Vinf, vorticity):
    return -2 * vorticity / Vinf

# Function to calculate upper and lower surface pressure coefficients
def Cp_upper_and_lower(Vinf, Pinf, rho, vorticity):
    P_u = 0.5 * rho * (Vinf - vorticity / 2)
    P_l = 0.5 * rho * (Vinf + vorticity / 2)

    Cp_u = (P_u - Pinf) / (0.5 * Vinf * Vinf * rho)
    Cp_l = (P_l - Pinf) / (0.5 * Vinf * Vinf * rho)

    return Cp_u, Cp_l

# Function to plot lift coefficient against angle of attack
def plot_Cl_alpha(AoA_range, chord, n_sections, Vinf, rho):
    thetas = theta_discr(n_sections)  # Discretize theta
    thetas_new = theta2(thetas)  # Get new theta values

    Cl_range = []  # Initialize list to store lift coefficients

    # Calculate lift coefficient for each angle of attack
    for AoA in AoA_range:
        vorticities = get_local_vorticity(AoA, Vinf, thetas_new)  # Calculate local vorticity
        lift = lift_calc(rho, Vinf, vorticities, chord, thetas_new)  # Calculate lift
        Cl_range.append(lift[1])  # Append lift coefficient to list

    ref_Clalpha = 2 * np.pi * AoA_range  # Reference lift coefficient
    delta = Cl_range - ref_Clalpha  # Difference between calculated and reference lift coefficient
    AoA_range = np.rad2deg(AoA_range)  # Convert angle of attack range to degrees

    # Plot lift coefficient against angle of attack
    plt.plot(AoA_range, Cl_range, label="$\mathregular{C_{l}}$ using thin-airfoil theory")
    plt.plot(AoA_range, ref_Clalpha, label="Reference $\mathregular{C_{l}}$ with slope of 2$\pi$")
    plt.plot(AoA_range, delta, label="Delta between approximation and reference")
    plt.xlabel(r'$ \alpha $ [deg]')
    plt.ylabel("$\mathregular{C_{l}}$ [-]")
    plt.grid()
    plt.legend()
    plt.show()

# Function to plot moment coefficient against angle of attack
def plot_Cm_alpha(AoA_range, chord, n_sections, Vinf, rho):
    thetas = theta_discr(n_sections)  # Discretize theta
    thetas_new = theta2(thetas)  # Get new theta values

    Cm_range = []  # Initialize list to store moment coefficients

    # Calculate moment coefficient for each angle of attack
    for AoA in AoA_range:
        vorticities = get_local_vorticity(AoA, Vinf, thetas_new)  # Calculate local vorticity
        moments = moment_calc(rho, Vinf, vorticities, chord, thetas_new)  # Calculate moments
        Cm_range.append(moments[1])  # Append moment coefficient to list

    ref_Cmalpha = -0.5 * np.pi * AoA_range  # Reference moment coefficient
    delta = Cm_range - ref_Cmalpha  # Difference between calculated and reference moment coefficient
    AoA_range = np.rad2deg(AoA_range)  # Convert angle of attack range to degrees

    # Plot moment coefficient against angle of attack
    plt.plot(AoA_range, Cm_range, label="$\mathregular{C_{m_{LE}}}$ using thin-airfoil theory")
    plt.plot(AoA_range, ref_Cmalpha, label="Reference $\mathregular{C_{m_{LE}}}$ with slope of -0.5$\pi$")
    plt.plot(AoA_range, delta, label="Delta between approximation and reference")
    plt.xlabel(r'$ \alpha $ [deg]')
    plt.ylabel("$\mathregular{C_{m_{LE}}}$ [-]")
    plt.grid()
    plt.legend()
    plt.show()

# Function to plot pressure coefficient against chordwise position
def plot_Cp(AoA, chord, n_sections, Vinf):
    thetas = theta_discr(n_sections)  # Discretize theta
    thetas_new = theta2(thetas)  # Get new theta values
    chord_discr = x_discr(thetas_new, chord)  # Discretize chord
    vorticities = get_local_vorticity(AoA, Vinf, thetas_new)  # Calculate local vorticity
    delta_Cp = delta_Cp_calc(Vinf, vorticities)  # Calculate pressure coefficient difference
    AoA_deg = np.rad2deg(AoA)  # Convert angle of attack to degrees

    # Plot pressure coefficient difference against chordwise position
    plt.plot(chord_discr, delta_Cp, label=r' Code AoA = {:0.2f} [-]'.format(AoA_deg), color='red')

# Function to read Xfoil file
def read_xfoil_file(name):
    f = open(name, 'r')  # Open file
    lines = f.readlines()  # Read lines from file
    dataupper = []  # Initialize list to store upper surface data
    datalower = []  # Initialize list to store lower surface data

    # Parse lines from file
    for l in lines[1:]:
        values = l.split()  # Split line into values
        if len(values) == 3:  # If line contains 3 values
            if float(values[1]) > 0:  # If second value is positive
                dataupper.append([float(val) for val in values])  # Append values to upper surface data list
            elif float(values[1]) < 0:  # If second value is negative
                datalower.append([float(val) for val in values])  # Append values to lower surface data list
    f.close()  # Close file

    dataupper_array = np.array(dataupper)  # Convert upper surface data list to array
    datalower_array = np.array(datalower)  # Convert lower surface data list to array

    return dataupper_array, datalower_array  # Return upper and lower surface data arrays

# Function to plot Xfoil data
def plot_xfoil_data(AoA_deg, xfoil_upper, xfoil_lower):
    deltaCp_data = []  # Initialize list to store pressure coefficient differences
    chord_distr = []  # Initialize list to store chordwise positions

    # Calculate pressure coefficient difference for each point on the upper surface
    for line in xfoil_upper:
        x = line[0]  # Chordwise position
        y = line[1]  # Vertical position (not used)
        Cp_upper = line[2]  # Pressure coefficient on upper surface

        # Find corresponding point on lower surface
        index = np.where(xfoil_lower[:, 0] == x)
        if index[0].size > 0:  # If corresponding point is found
            Cp_lower = xfoil_lower[index, 2][0]  # Pressure coefficient on lower surface

        deltaCp = Cp_upper - Cp_lower  # Pressure coefficient difference
        deltaCp_data.append(deltaCp)  # Append pressure coefficient difference to list
        chord_distr.append(x)  # Append chordwise position to list

    # Plot pressure coefficient difference against chordwise position
    plt.plot(chord_distr, deltaCp_data, label=r'NACA0001 Xfoil AoA = {:0.2f} [-]'.format(AoA_deg), color='blue')

    return dataupper_array, datalower_array

# Main function
if __name__ == "__main__":
    max_AoA_deg = (input('Max Angle of Attack [deg]? '))  # Get maximum angle of attack from user

    thetas = theta_discr(n_sections)  # Discretize theta
    thetas_new = theta2(thetas)  # Get new theta values

    # Calculate lift and moment coefficients for each angle of attack
    for AoA_deg in range(1, int(max_AoA_deg) + 1, 1):
        AoA = np.deg2rad(AoA_deg)  # Convert angle of attack to radians
        vorticities = get_local_vorticity(AoA, Vinf, thetas_new)  # Calculate local vorticity
        C_l = lift_calc(rho, Vinf, vorticities, chord, thetas_new)  # Calculate lift coefficient
        C_m = moment_calc(rho, Vinf, vorticities, chord, thetas_new)  # Calculate moment coefficient

        print('Lift coefficient C_l: ', C_l[1])  # Print lift coefficient
        print('Moment coefficient C_m: ', C_m[1])  # Print moment coefficient

        # Read Xfoil data from file
        dataupper_array, datalower_array = read_xfoil_file(f'cp-aoa{AoA_deg}')

        # Plot Xfoil data
        plot_xfoil_data(AoA_deg, dataupper_array, datalower_array)
        plot_Cp(AoA, chord, n_sections, Vinf)

    # Set plot labels and properties
    plt.xlabel("x [m]")
    plt.ylabel("$\Delta\mathregular{C_{p}}$ [-]")
    plt.gca().invert_yaxis()
    plt.ylim((1, -4))
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()

    # Plot pressure coefficient difference for experimental data
    plot_Cp(exp_AoA, chord, n_sections, Vinf)
    plt.plot(exp_data_chord, exp_data_deltacp_4, label=r'NACA0006 Exp. AoA = {:0.2f} [-]'.format(np.rad2deg(exp_AoA)))

    # Set plot labels and properties
    plt.xlabel("x [m]")
    plt.ylabel("$\Delta\mathregular{C_{p}}$ [-]")
    plt.gca().invert_yaxis()
    plt.ylim((1, -4))
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
