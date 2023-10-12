import matplotlib.pyplot as plt
import numpy as np

import forcesmoments as fm
import vorticity as vort


def plot_Cl_alpha(AoA_range, chord, n_sections, Vinf, rho):
    thetas = vort.get_theta_from_x(chord, n_sections)[0]

    Cl_range = []

    for AoA in AoA_range:
        vorticities = vort.get_local_vorticity(AoA, Vinf, thetas)
        lift = fm.lift(rho, Vinf, vorticities, chord, thetas)
        Cl_range.append(lift[1])

    ref_Clalpha = 2 * np.pi * AoA_range
    plt.plot(AoA_range, Cl_range, label="calculated")
    plt.plot(AoA_range, ref_Clalpha, label="2pi")
    plt.legend()
    plt.show()

def plot_Cm_alpha(AoA_range, chord, n_sections, Vinf, rho):
    thetas = vort.get_theta_from_x(chord, n_sections)[0]

    Cm_range = []

    for AoA in AoA_range:
        vorticities = vort.get_local_vorticity(AoA, Vinf, thetas)
        moments = fm.moments(rho, Vinf, vorticities, chord, thetas)
        Cm_range.append(moments[1])

    ref_Cmalpha = -0.5*np.pi*AoA_range
    plt.plot(AoA_range, Cm_range, label="calculated")
    plt.plot(AoA_range, ref_Cmalpha, label="-0.5pi")
    plt.legend()
    plt.show()

def plot_Cp(AoA, chord, n_sections, Vinf):
    thetas, chord_discr = vort.get_theta_from_x(chord, n_sections)
    vorticities = vort.get_local_vorticity(AoA, Vinf, thetas)
    delta_Cp = fm.delta_Cp(Vinf, vorticities)
    plt.plot(chord_discr, delta_Cp, label="Cp distribution")
    plt.legend()
    plt.show()
