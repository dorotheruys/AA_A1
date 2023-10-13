import numpy as np


def lift(rho, Vinf, vorticity, chord, thetas):
    # Inputs:
    # Vorticity distribution, Vinf, rho, chord, thetas
    lprime = 0.5 * chord * rho * Vinf * np.trapz(vorticity * np.sin(thetas), thetas)
    C_l = lprime / (0.5 * rho * Vinf * Vinf * chord)
    return lprime, C_l


def moments(rho, Vinf, vorticity, chord, thetas):
    Mprime_LE = -0.25 * chord * chord * rho * Vinf * np.trapz(vorticity * (1 - np.cos(thetas)) * np.sin(thetas), thetas)
    C_M_LE = Mprime_LE / (0.5 * rho * Vinf * Vinf * chord * chord)
    return Mprime_LE, C_M_LE


def delta_Cp(Vinf, vorticity):
    return 2*vorticity/Vinf
