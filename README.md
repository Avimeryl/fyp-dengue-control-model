
# Dengue Fever Control Simulation and Model Fitting

This project models and simulates the dynamics of dengue fever transmission in Sarawak, Malaysia, using the ASEI-SVIR model with optimal control strategies. The project consists of two main parts:

1. **Model Fitting (`ModelFitting.py`)**: Fits the dengue transmission model to real dengue case data using nonlinear least squares.
2. **Optimal Control Simulation (`ModelSimulation/`)**: Implements a forward-backward sweep method to evaluate the effectiveness of three control strategies:
    - `u1`: Larvicide (pre-adult mosquito control)
    - `u2`: Insecticide/fogging (adult mosquito control)
    - `u3`: Vaccination (human immunity)

---

## 📁 Project Structure

```
├── ModelFitting.py                  # Parameter fitting using cumulative real data
├── ModelSimulation/
│   ├── Main.py                      # Main simulation for optimal control strategies
│   ├── State_Function.py            # Defines the differential equations
│   ├── Adjoint_Function.py          # Adjoint equations for backward sweep
│   ├── Control_Function.py          # Control update function
│   ├── Plot_Graph.py                # Plotting utilities
```

---

## 📊 Dataset

Weekly dengue fever cases in Malaysia for 41 weeks:
- Used as input for model fitting (`ModelFitting.py`)
- Converted to cumulative values for parameter estimation

---

## 🔧 How to Run

### 1. Fit the Model to Real Data
```bash
python ModelFitting.py
```
- Produces a plot comparing observed and predicted cumulative infected cases
- Outputs the optimized parameter values

### 2. Simulate Control Strategies
```bash
cd ModelSimulation
python Main.py
```
- Simulates 7 strategies (including individual and combined controls)
- Displays infection trends in both humans and mosquitoes
- Visualizes control values (`u1`, `u2`, `u3`) over time

---

## 📈 Control Strategies Overview

| Strategy | u1 (Larvicide) | u2 (Fogging) | u3 (Vaccination) |
|----------|----------------|--------------|------------------|
| 1        | ✅              | ❌            | ❌                |
| 2        | ❌              | ✅            | ❌                |
| 3        | ❌              | ❌            | ✅                |
| 4        | ✅              | ✅            | ❌                |
| 5        | ✅              | ❌            | ✅                |
| 6        | ❌              | ✅            | ✅                |
| 7        | ✅              | ✅            | ✅                |

---

## 📌 Key Features

- Realistic simulation using fitted parameters
- Forward-Backward Sweep Method with Runge-Kutta 4th Order
- Costate-based optimal control updates
- Visualization of infection curves and control efforts

---

## 📬 Contact

Project Title: _Evaluating the Effectiveness of Intervention Strategies in Controlling Dengue Fever Outbreaks with ASEI-SVIR Model_

Ibnu Ameerul Bin Abdul Halim

Bachelor of Computer Science (Computational Science) with Honors

Universiti Malaysia Sarawak

📧 ameerulibnu69@gmail.com

