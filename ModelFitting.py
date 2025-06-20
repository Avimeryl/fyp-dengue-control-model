import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# Define the differential equations for the model
def mosquito_human_model(t, y, φ, k, ω, B, β_h, β_v, δ):
    A_v, S_v, E_v, I_v, S_h, V_h, I_h, R_h = y
    N_h = 34058800  # Total human population
    μ_A = 0.42      # Natural mortality of larvae (per week)
    μ_h = 0.0069    # Natural mortality rate of humans (per week)
    θ = 0.0018      # Vaccine waning rate (per week)
    σ = 0.198       # Proportion of vaccinated humans infected
    γ = 0.7         # Human recovery rate (per week)
    μ_v = 0.7       # Natural mortality of mosquitoes (per week)

    # Mosquito equations
    dA_v = φ * (1 - A_v / (k * N_h)) * (S_v + E_v + I_v) - (ω + μ_A) * A_v
    dS_v = ω * A_v - (B * β_v * I_h / N_h + μ_v) * S_v
    dE_v = (B * β_v * I_h / N_h) * S_v - (δ + μ_v) * E_v
    dI_v = δ * E_v - μ_v * I_v

    # Human equations
    dS_h = μ_h * N_h + θ * V_h - (B * β_h * I_v / N_h + μ_h) * S_h
    dV_h = -(θ + σ * B * β_h * I_v / N_h + μ_h) * V_h
    dI_h = (B * β_h * I_v / N_h) * S_h + σ * (B * β_h * I_v / N_h) * V_h - (γ + μ_h) * I_h
    dR_h = γ * I_h - μ_h * R_h

    return [dA_v, dS_v, dE_v, dI_v, dS_h, dV_h, dI_h, dR_h]

# Function to solve the ODE system
def solve_model(week, φ, k, ω, B, β_h, β_v, δ):
    # Initial conditions
    N_h = 34058800
    S_h0 = N_h - 3181
    V_h0 = 0
    I_h0 = 3181
    R_h0 = 0
    A_v0 = 3 * N_h
    S_v0 = (4 * N_h) - 200000
    E_v0 = 100000
    I_v0 = 100000
    y0 = [A_v0, S_v0, E_v0, I_v0, S_h0, V_h0, I_h0, R_h0]
    
    # Time vector
    t = np.linspace(0, week[-1], len(week))
    
    # Solve the system using solve_ivp
    sol = solve_ivp(
        lambda t, y: mosquito_human_model(t, y, φ, k, ω, B, β_h, β_v, δ),
        [0, t[-1]], y0, t_eval=t, method="RK45"
    )
    
    return sol.t, sol.y[6]  # Return time and infected humans (I_h)

# Load data for weeks and number of cases
weeks = np.arange(1, 42)  # Week numbers
cases = np.array([3181, 3525, 3971, 3781, 3969, 3631, 3483, 3572, 3268, 3238,
                  2905, 3041, 2579, 2487, 1698, 2321, 2237, 1995, 2338, 2461,
                  2426, 2522, 2508, 2900, 2438, 2788, 2805, 2373, 2690, 2609,
                  2515, 2278, 2294, 1980, 1765, 1870, 1794, 1514, 1760, 1639, 1624])

# Compute cumulative real infected cases
cumulative_cases = np.cumsum(cases)

# Define a wrapper function for curve fitting
def fit_function(week, φ, k, ω, B, β_h, β_v, δ):
    t, I_h = solve_model(week, φ, k, ω, B, β_h, β_v, δ)
    cumulative_I_h = np.cumsum(I_h)  # Compute cumulative infected humans
    return cumulative_I_h

# Adjusted initial parameter guesses
initial_guess = [7.5, 35, 0.75, 0.78, 0.15, 0.65, 0.55]  # Modify as needed

# Adjusted parameter bounds
param_bounds = (
    [5.1, 25.5, 0.51, 0.71, 0.11, 0.51, 0.51],  # Lower bounds
    [9.9, 39.9, 0.99, 0.89, 0.99, 0.89, 0.89]   # Upper bounds
)

# Modify initial conditions for mosquitoes and humans
def solve_model(week, φ, k, ω, B, β_h, β_v, δ):
    N_h = 34058800  # Total human population
    # Adjust initial conditions for better early-week dynamics
    S_h0 = N_h - 3000  # Modify as needed
    I_h0 = 3000  # Modify as needed
    R_h0 = 0
    A_v0 = 2.5 * N_h  # Adjust mosquito larvae population
    S_v0 = (3.5 * N_h) - 150000  # Modify mosquito susceptible population
    E_v0 = 80000  # Modify mosquito exposed population
    I_v0 = 90000  # Modify mosquito infectious population
    V_h0 = 0  # Vaccinated humans remain 0 initially

    y0 = [A_v0, S_v0, E_v0, I_v0, S_h0, V_h0, I_h0, R_h0]

    t = np.linspace(0, week[-1], len(week))
    
    sol = solve_ivp(
        lambda t, y: mosquito_human_model(t, y, φ, k, ω, B, β_h, β_v, δ),
        [0, t[-1]], y0, t_eval=t, method="RK45"
    )
    
    return sol.t, sol.y[6]

# Define a wrapper function for curve fitting
def fit_function(week, φ, k, ω, B, β_h, β_v, δ):
    t, I_h = solve_model(week, φ, k, ω, B, β_h, β_v, δ)
    cumulative_I_h = np.cumsum(I_h)
    return cumulative_I_h

# Adjust fitting to prioritize earlier weeks (optional)
weights = np.concatenate([np.ones(15) * 2, np.ones(len(weeks) - 15)])  # Double weight for weeks 1-15

# Curve fitting
popt, pcov = curve_fit(
    fit_function, weeks, cumulative_cases, p0=initial_guess, bounds=param_bounds, sigma=1 / weights, maxfev=30000
)

# Extract fitted parameters
φ_fit, k_fit, ω_fit, B_fit, β_h_fit, β_v_fit, δ_fit = popt

# Solve model with fitted parameters
t_fitted, fitted_I_h = solve_model(weeks, φ_fit, k_fit, ω_fit, B_fit, β_h_fit, β_v_fit, δ_fit)
fitted_cumulative_cases = np.cumsum(fitted_I_h)

# Plot cumulative cases
plt.figure(figsize=(10, 6))
plt.scatter(weeks, cumulative_cases, label="Observed Cumulative Cases", color="red")
plt.plot(weeks, fitted_cumulative_cases, label="Fitted Model (Cumulative)", color="blue")
plt.xlabel("Week")
plt.ylabel("Cumulative Cases (I_h)")
plt.title("Cumulative Infected Cases vs. Time")
plt.legend()
plt.grid()
plt.show()

# Print fitted parameters
print("Fitted Parameters:")
print(f"φ = {φ_fit:.6f}, k = {k_fit:.6f}, ω = {ω_fit:.6f}, B = {B_fit:.6f}")
print(f"β_h = {β_h_fit:.6f}, β_v = {β_v_fit:.6f}, δ = {δ_fit:.6f}")
