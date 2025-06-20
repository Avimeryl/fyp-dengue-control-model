import numpy as np

def adjoint(t, lam, y, u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh):
    Av, Sv, Ev, Iv, Sh, Vh, Ih, Rh = y
    lambda_Av, lambda_Sv, lambda_Ev, lambda_Iv, lambda_Sh, lambda_Vh, lambda_Ih, lambda_Rh = lam

    # Weight constants
    W1 = 100            # Iv
    W2 = 100            # Ih
    W3 = 1000000000     # larvicide
    W4 = 2500000000     # insecticide
    W5 = 100000000000   # vaccination

    # Adjoint equations
    d_lambda_Av = -(lambda_Av * ((-(Sv + Ev + Iv)/(k * Nh)) - omega - muA - u1) + lambda_Sv * omega)
    d_lambda_Sv = -(lambda_Sv * (-((B * Beta_v * Ih) / Nh) - muv - u2) + lambda_Av * phi * (1 - (Av / (k * Nh))) + lambda_Ev * (B * Beta_v * Ih / Nh))
    d_lambda_Ev = -(lambda_Ev * (- delta - muv - u2) + lambda_Av * phi * (1 - (Av / (k * Nh))) + lambda_Iv * delta)
    d_lambda_Iv = -(lambda_Iv * (- muv - u2) + lambda_Av * phi * (1 - (Av / (k * Nh))) + lambda_Sh * (B * Beta_h * Sh / Nh) -  lambda_Vh * (sigma * B * Beta_h * Vh / Nh) + lambda_Ih * ((B * Beta_h * Sh / Nh) + (sigma * B * Beta_h * Vh / Nh)) + W1)
    d_lambda_Sh = -(lambda_Sh * (-(B * Beta_h * Iv / Nh) - muh - u3) + lambda_Vh * u3 + lambda_Ih * (B * Beta_h * Iv / Nh))
    d_lambda_Vh = -(lambda_Vh * (- theta - (sigma * B * Beta_h * Iv / Nh) - muh) + (lambda_Sh * theta) +  lambda_Ih * (sigma * B * Beta_h * Iv / Nh))
    d_lambda_Ih = -(lambda_Ih * (- gamma + muh) + (lambda_Rh * gamma) - lambda_Sv * (B * Beta_v * Sv / Nh) + lambda_Ev * (B * Beta_v * Sv / Nh) + W2)
    d_lambda_Rh = lambda_Rh * muh

    return np.array([d_lambda_Av, d_lambda_Sv, d_lambda_Ev, d_lambda_Iv, d_lambda_Sh, d_lambda_Vh, d_lambda_Ih, d_lambda_Rh])