import numpy as np


def update_controls(y, lam, W3, W4, W5, u1_prev, u2_prev, u3_prev, alpha=0.5):  
    Av, Sv, Ev, Iv, Sh, Vh, Ih, Rh = y
    lambda_Av, lambda_Sv, lambda_Ev, lambda_Iv, lambda_Sh, lambda_Vh, lambda_Ih, lambda_Rh = lam

    # Compute new control values
    computed_u1 = np.clip((lambda_Av * Av) / W3, 0, 1)  
    computed_u2 = np.clip(((lambda_Sv * Sv) + (lambda_Ev * Ev) + (lambda_Iv * Iv)) / W4, 0, 1)
    computed_u3 = np.clip(((lambda_Sh * Sh) + (lambda_Vh * Sh)) / W5, 0, 1)

    # Apply relaxation factor to smooth updates
    u1 = (1 - alpha) * u1_prev + alpha * computed_u1
    u2 = (1 - alpha) * u2_prev + alpha * computed_u2
    u3 = (1 - alpha) * u3_prev + alpha * computed_u3

    # Ensure values remain within [0,1]
    u1 = np.clip(u1, 0, 1)
    u2 = np.clip(u2, 0, 1)
    u3 = np.clip(u3, 0, 1)

    return u1, u2, u3