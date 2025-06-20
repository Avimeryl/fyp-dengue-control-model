import numpy as np
import matplotlib.pyplot as plt
from State_Function import mosquito_human_model
from Adjoint_Function import adjoint
from Control_Function import update_controls
from Plot_Graph import plot_strategy, plot_controls

# Time parameters
t0 = 0
T = 42              # Total time in weeks
N = 100             # Number of time steps
dt = (T - t0) / N

# Initialize state and adjoint variables
y = np.zeros((N + 1, 8))    # State variables
lam = np.zeros((N + 1, 8))  # Adjoint variables

#parameters value
Nh = 34058800     # Total human population
muA = 0.42        # Natural mortality of larvae (per week)
muh = 0.0069      # Natural mortality rate of humans (per week)
theta = 0.0018    # Vaccine waning rate (per week)
sigma = 0.198     # Proportion of vaccinated humans infected
gamma = 0.7       # Human recovery rate (per week)
muv = 0.7         # Natural mortality of mosquitoes (per week)
phi = 5.1         # Oviposition rate 
k = 39.9          # Mosquito larvae density per person
omega = 0.897548  # Larvae development rate to adult
B = 0.71          # Mosquito bite rate
Beta_h = 0.11     # Infection transmission rate from infected mosquitoes to susceptible humans
Beta_v = 0.51     # Infection transmission rate from infected humans to susceptible mosquitoes
delta = 0.51      # Rate of disease advancement in exposed mosquitoes


# Initial conditions (same as defined in State_Function.py)
Sh0 = 34058800 - 3181
Vh0 = 0
Ih0 = 3181
Rh0 = 0
Av0 = 3 * 34058800
Sv0 = (4 * 34058800) - 200000
Ev0 = 100000
Iv0 = 100000
y[0] = [Av0, Sv0, Ev0, Iv0, Sh0, Vh0, Ih0, Rh0]

# Initial controls
u1, u2, u3 = 0.001, 0.001, 0.001

# Weights for control
W3, W4, W5 = 1e9, 2.5e9, 1e11

# RK4 Forward function
def RK4_forward(y, u1, u2, u3):
    for i in range(N):
        k1 = np.array(mosquito_human_model(i * dt, y[i], u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh))
        k2 = np.array(mosquito_human_model((i + 0.5) * dt, y[i] + 0.5 * dt * k1, u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh))
        k3 = np.array(mosquito_human_model((i + 0.5) * dt, y[i] + 0.5 * dt * k2, u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh))
        k4 = np.array(mosquito_human_model((i + 1) * dt, y[i] + dt * k3, u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh))
        y[i + 1] = y[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

# RK4 Backward function
def RK4_backward(lam, y, u1, u2, u3):
    lam[-1] = np.zeros(8)  # Final boundary conditions
    for i in range(N - 1, -1, -1):
        k1 = np.array(adjoint(i * dt, lam[i], y[i], u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh))
        k2 = np.array(adjoint((i - 0.5) * dt, lam[i] - 0.5 * dt * k1, y[i], u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh))
        k3 = np.array(adjoint((i - 0.5) * dt, lam[i] - 0.5 * dt * k2, y[i], u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh))
        k4 = np.array(adjoint((i - 1) * dt, lam[i] - dt * k3, y[i], u1, u2, u3, phi, muA, muh, muv, k, omega, B, Beta_h, Beta_v, delta, theta, sigma, gamma, Nh))
        lam[i - 1] = lam[i] - (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return lam

# Initialize previous control variables for relaxation
u1_prev, u2_prev, u3_prev = 0.001, 0.001, 0.001  # Small initial values

# Define relaxation factor (controls smoothness)
alpha = 0.1  # Adjust this value between 0 and 1 for more gradual updates

# Arrays to store control values over time
u1_values, u2_values, u3_values = [], [], []

# Forward-Backward Sweep Method
for _ in range(100):                                       # Iterate to find optimal control
    y = RK4_forward(y, u1_prev, u2_prev, u3_prev)          # Forward simulation
    lam = RK4_backward(lam, y, u1_prev, u2_prev, u3_prev)  # Backward adjoint

    # Compute new control values based on costates (lambda)
    u1_new, u2_new, u3_new = update_controls(y[-1], lam[-1], W3, W4, W5, u1_prev, u2_prev, u3_prev)

    # Apply relaxation for smooth transitions
    u1_prev = (1 - alpha) * u1_prev + alpha * u1_new
    u2_prev = (1 - alpha) * u2_prev + alpha * u2_new
    u3_prev = (1 - alpha) * u3_prev + alpha * u3_new

    # Store control values for plotting
    u1_values.append(u1_prev)
    u2_values.append(u2_prev)
    u3_values.append(u3_prev)

# Convert lists to numpy arrays for plotting
u1_values = np.array(u1_values)
u2_values = np.array(u2_values)
u3_values = np.array(u3_values)


# Time array
time = np.linspace(t0, T, N + 1)

# Run simulation with u1 = 0, u2 = 0, u3 = 0
u1_zero, u2_zero, u3_zero = 0, 0, 0
y_zero_control = RK4_forward(np.copy(y), u1_zero, u2_zero, u3_zero)

# Extract Ih and Iv for no control scenario 
Av_zero_control = y_zero_control[:, 0] 
Sv_zero_control = y_zero_control[:, 1]    
Ev_zero_control = y_zero_control[:, 2]  
Iv_zero_control = y_zero_control[:, 3] 
Sh_zero_control = y_zero_control[:, 4] 
Vh_zero_control = y_zero_control[:, 5]
Ih_zero_control = y_zero_control[:, 6]  
Rh_zero_control = y_zero_control[:, 7]     

# Strategy 1: u1 ≠ 0, u2 = 0, u3 = 0
y_u1_control = RK4_forward(np.copy(y), u1_prev, 0, 0)
Ih_u1_control = y_u1_control[:, 6]  
Iv_u1_control = y_u1_control[:, 3] 

# Strategy 2: u1 = 0, u2 ≠ 0, u3 = 0
y_u2_control = RK4_forward(np.copy(y), 0, u2_prev, 0)
Ih_u2_control = y_u2_control[:, 6]  
Iv_u2_control = y_u2_control[:, 3]  

# Strategy 3: u1 = 0, u2 = 0, u3 ≠ 0
y_u3_control = RK4_forward(np.copy(y), 0, 0, u3_prev)
Ih_u3_control = y_u3_control[:, 6]  
Iv_u3_control = y_u3_control[:, 3] 

# Strategy 4: u1 ≠ 0, u2 ≠ 0, u3 = 0
y_u1_u2_control = RK4_forward(np.copy(y), u1_prev, u2_prev, 0)
Ih_u1_u2_control = y_u1_u2_control[:, 6]
Iv_u1_u2_control = y_u1_u2_control[:, 3]

# Strategy 5: u1 ≠ 0, u2 = 0, u3 ≠ 0
y_u1_u3_control = RK4_forward(np.copy(y), u1_prev, 0, u3_prev)
Ih_u1_u3_control = y_u1_u3_control[:, 6]
Iv_u1_u3_control = y_u1_u3_control[:, 3]

# Strategy 6: u1 = 0, u2 ≠ 0, u3 ≠ 0
y_u2_u3_control = RK4_forward(np.copy(y), 0, u2_prev, u3_prev)
Ih_u2_u3_control = y_u2_u3_control[:, 6]
Iv_u2_u3_control = y_u2_u3_control[:, 3]

# Strategy 7: u1 ≠ 0, u2 ≠ 0, u3 ≠ 0
y_u1_u2_u3_control = RK4_forward(np.copy(y), u1_prev, u2_prev, u3_prev)
Av_u1_u2_u3_control = y_u1_u2_u3_control[:, 0]
Sv_u1_u2_u3_control = y_u1_u2_u3_control[:, 1]
Ev_u1_u2_u3_control = y_u1_u2_u3_control[:, 2]
Iv_u1_u2_u3_control = y_u1_u2_u3_control[:, 3]
Sh_u1_u2_u3_control = y_u1_u2_u3_control[:, 4]
Vh_u1_u2_u3_control = y_u1_u2_u3_control[:, 5]
Ih_u1_u2_u3_control = y_u1_u2_u3_control[:, 6]
Rh_u1_u2_u3_control = y_u1_u2_u3_control[:, 7]

# Plot for Strategy 1 - Human Infections
plot_strategy(
    time,
    Ih_u1_control,
    Ih_zero_control,
    label_control='With Control (u1≠0, u2=0, u3=0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 1: Effect of u1 on Infected Human Population',
    ylabel='Infected Humans (Ih)'
)

# Plot for Strategy 1 - Vector Infections
plot_strategy(
    time,
    Iv_u1_control,
    Iv_zero_control,
    label_control='With Control (u1≠0, u2=0, u3=0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 1: Effect of u1 on Infected Vector Population',
    ylabel='Infected Vectors (Iv)'
)

# Strategy 1: Only u1
plot_controls(time[:-1], u1=u1_values, title='Control Strategy 1 (u1 only)')


# Plot for Strategy 2 - Human Infections
plot_strategy(
    time,
    Ih_u2_control,
    Ih_zero_control,
    label_control='With Control (u1=0, u2≠0, u3=0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 2: Effect of u2 on Infected Human Population',
    ylabel='Infected Humans (Ih)'
)

# Plot for Strategy 2 - Vector Infections
plot_strategy(
    time,
    Iv_u2_control,
    Iv_zero_control,
    label_control='With Control (u1=0, u2≠0, u3=0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 2: Effect of u2 on Infected Vector Population',
    ylabel='Infected Vectors (Iv)'
)

# Strategy 2: Only u2
plot_controls(time[:-1], u2=u2_values, title='Control Strategy 2 (u2 only)')

# Plot for Strategy 3 - Human Infections
plot_strategy(
    time,
    Ih_u3_control,
    Ih_zero_control,
    label_control='With Control (u1=0, u2=0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 3: Effect of u3 on Infected Human Population',
    ylabel='Infected Humans (Ih)'
)

# Plot for Strategy 3 - Vector Infections
plot_strategy(
    time,
    Iv_u3_control,
    Iv_zero_control,
    label_control='With Control (u1=0, u2=0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 3: Effect of u3 on Infected Vector Population',
    ylabel='Infected Vectors (Iv)'
)

# Strategy 3: Only u3
plot_controls(time[:-1], u3=u3_values, title='Control Strategy 3 (u3 only)')

# Plot for Strategy 4 - Human Infections
plot_strategy(
    time,
    Ih_u1_u2_control,
    Ih_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3=0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 4: Effect of u1 & u2 on Infected Human Population',
    ylabel='Infected Humans (Ih)'
)

# Plot for Strategy 4 - Vector Infections
plot_strategy(
    time,
    Iv_u1_u2_control,
    Iv_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3=0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 4: Effect of u1 & u2 on Infected Vector Population',
    ylabel='Infected Vectors (Iv)'
)

# Strategy 4: u1 and u2
plot_controls(time[:-1], u1=u1_values, u2=u2_values, title='Control Strategy 4 (u1 & u2)')

# Plot for Strategy 5 - Human Infections
plot_strategy(
    time,
    Ih_u1_u3_control,
    Ih_zero_control,
    label_control='With Control (u1≠0, u2=0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 5: Effect of u1 & u3 on Infected Human Population',
    ylabel='Infected Humans (Ih)'
)

# Plot for Strategy 5 - Vector Infections
plot_strategy(
    time,
    Iv_u1_u3_control,
    Iv_zero_control,
    label_control='With Control (u1≠0, u2=0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 5: Effect of u1 & u3 on Infected Vector Population',
    ylabel='Infected Vectors (Iv)'
)

# Strategy 5: u1 and u3
plot_controls(time[:-1], u1=u1_values, u3=u3_values, title='Control Strategy 5 (u1 & u3)')

# Plot for Strategy 6 - Human Infections
plot_strategy(
    time,
    Ih_u2_u3_control,
    Ih_zero_control,
    label_control='With Control (u1=0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 6: Effect of u2 & u3 on Infected Human Population',
    ylabel='Infected Humans (Ih)'
)

# Plot for Strategy 6 - Vector Infections
plot_strategy(
    time,
    Iv_u2_u3_control,
    Iv_zero_control,
    label_control='With Control (u1=0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 6: Effect of u2 & u3 on Infected Vector Population',
    ylabel='Infected Vectors (Iv)'
)

# Strategy 6: u2 and u3
plot_controls(time[:-1], u2=u2_values, u3=u3_values, title='Control Strategy 6 (u2 & u3)')

# Plot for Strategy 7 - Pre-Adult Mosquito
plot_strategy(
    time,
    Av_u1_u2_u3_control,
    Av_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 7: Effect of u1, u2 & u3 on Pre-Adult Mosquito Population',
    ylabel='Pre-Adult Mosquito (Av)'
)

# Plot for Strategy 7 - Susceptible Mosquito
plot_strategy(
    time,
    Sv_u1_u2_u3_control,
    Sv_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 7: Effect of u1, u2 & u3 on Susceptible Mosquito Population',
    ylabel='Susceptible (Sv)'
)

# Plot for Strategy 7 - Exposed Mosquito
plot_strategy(
    time,
    Ev_u1_u2_u3_control,
    Ev_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 7: Effect of u1, u2 & u3 on Exposed Mosquito Population',
    ylabel='Exposed (Ev)'
)

# Plot for Strategy 7 - Vector Infections
plot_strategy(
    time,
    Iv_u1_u2_u3_control,
    Iv_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 7: Effect of u1, u2 & u3 on Infected Vector Population',
    ylabel='Infected Vectors (Iv)'
)

# Plot for Strategy 7 - Susceptible Human
plot_strategy(
    time,
    Sh_u1_u2_u3_control,
    Sh_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 7: Effect of u1, u2 & u3 on Susceptible Human Population',
    ylabel='Susceptible Humans (Sh)'
)

# Plot for Strategy 7 - Vaccinated Human
plot_strategy(
    time,
    Vh_u1_u2_u3_control,
    Vh_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 7: Effect of u1, u2 & u3 on Vaccinated Human Population',
    ylabel='Vaccinated Humans (Vh)'
)

# Plot for Strategy 7 - Human Infections
plot_strategy(
    time,
    Ih_u1_u2_u3_control,
    Ih_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 7: Effect of u1, u2 & u3 on Infected Human Population',
    ylabel='Infected Humans (Ih)'
)

# Plot for Strategy 7 - Recovered Human
plot_strategy(
    time,
    Rh_u1_u2_u3_control,
    Rh_zero_control,
    label_control='With Control (u1≠0, u2≠0, u3≠0)',
    label_comparison='No Control (u1=0, u2=0, u3=0)',
    title='Strategy 7: Effect of u1, u2 & u3 on Recovered Human Population',
    ylabel='Recovered Humans (Rh)'
)

# Strategy 7: u1, u2, and u3
plot_controls(time[:-1], u1=u1_values, u2=u2_values, u3=u3_values, title='Control Strategy 7 (u1, u2 & u3)')