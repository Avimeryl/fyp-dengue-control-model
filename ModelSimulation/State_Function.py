import numpy as np

def mosquito_human_model(t, y, u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh):
    Av, Sv, Ev, Iv, Sh, Vh, Ih, Rh = y

    # Mosquito equations
    dAv_dt = phi * (1 - (Av / (k * Nh))) * (Sv + Ev + Iv) - (omega + muA + u1) * Av  # Pre-adult / Larva
    dSv_dt = omega * Av - ((B * (Beta_v * Ih / Nh)) + muv + u2) * Sv                 # Susceptible Mosquito
    dEv_dt = (B * (Beta_v * Ih / Nh) * Sv) - (delta + muv + u2) * Ev                 # Exposed Mosquito
    dIv_dt = (delta * Ev) - (muv + u2) * Iv                                          # Infected Mosquito

    # Human equations
    dSh_dt = (muh * Nh) + (theta * Vh) - ((B * (Beta_h * Iv / Nh)) + muh + u3) * Sh                      # Susceptible Human
    dVh_dt = (u3 * Sh) - (theta + ((sigma * B * Beta_h * Iv) / Nh) + muh) * Vh                           # Vaccinated Human
    dIh_dt = ((B * Sh * Beta_h * Iv) / Nh) + ((sigma * Vh * B * Beta_h * Iv) / Nh) - (gamma + muh) * Ih  # Infected Human
    dRh_dt = (gamma * Ih) - (muh * Rh)                                                                   # Recovered Human

    return np.array([dAv_dt, dSv_dt, dEv_dt, dIv_dt, dSh_dt, dVh_dt, dIh_dt, dRh_dt])