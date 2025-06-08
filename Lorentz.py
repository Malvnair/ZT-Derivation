import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Constants
k_B = 1.380649e-23        # Boltzmann constant, J/K
eV_to_J = 1.602176634e-19 # 1 eV in J
e = 1.602176634e-19       # Elementary charge, C
T = 300.0                 # Temperature, K

# Constant D: D = exp[-π^2 * k_B / (3 * e * 116)]
D = mp.e**(-mp.pi**2 * k_B / (3 * e * 116))

# Chemical potential range in eV
mu_eV = np.linspace(-0.5, 0.5, 200)

# Compute Lorenz number for each μ
L_vals = []
for mu in mu_eV:
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    z = -mp.e**(eta)
    Li_half  = mp.polylog(0.5, z).real
    Li_3half = mp.polylog(1.5, z).real
    R = abs(Li_half / Li_3half)              # Dimensionless ratio
    L_dimless = 1.5 + D**R                   # Dimensionless Lorenz number
    L_SI = L_dimless * 1e-8                  # Convert to W·Ω·K⁻²
    L_vals.append(float(L_SI))

L_vals = np.array(L_vals)

# Plotting
plt.plot(mu_eV, L_vals, linewidth=2)
plt.xlabel("μ (eV)")
plt.ylabel("L (W·Ω·K⁻²)")
plt.title("Lorenz Number Versus Chemical Potential at 300 K")
plt.grid(True)
plt.tight_layout()
plt.show()
